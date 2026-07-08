#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Measure the *actual* per-rotor update rate on /beetle_omni/esc_telem (ROS 1).

Why this is not just "1 / topic_hz":
  spinal/ESCTelemetryArray is published once per full DShot telemetry round-robin
  cycle (see attitude_control.cpp: published only when `is_update_all_msg_` is set,
  i.e. after all 4 ESCs have had their turn). Every publish carries all 4 rotor
  slots, but only the rotor whose turn it was this cycle actually got a fresh
  reading from esc_telem.cpp::update() - the other three slots are just copies of
  whatever they held from their own last turn. So the topic publish rate and the
  rate at which any single rotor's data actually refreshes are different things.

This script detects a "real update" for a rotor by diffing its physical fields
(temperature, voltage, current, consumption, rpm) against the previous message.
If none of them changed, the slot was not touched this cycle and is not counted.
crc_error is tracked separately (it is only meaningful when esc_telem.cpp actually
captured a full 10-byte frame; a nonzero value there means that frame failed CRC).

Usage:
  rosrun beetle_omni ESCTelemUpdateAnalysis.py
  python3 ESCTelemUpdateAnalysis.py _topic:=/beetle_omni/esc_telem _duration:=20
  python3 ESCTelemUpdateAnalysis.py _duration:=0   # run until Ctrl+C

Params (rosrun private, "_name:=value"):
  ~topic     (str)   default "/beetle_omni/esc_telem"
  ~duration  (float) default 20.0, seconds to record; <=0 means run until Ctrl+C
  ~save_csv  (bool)  default True, dump per-rotor update timestamps next to this
                      script as ESCTelemUpdateAnalysis_<stamp>.csv
"""

import csv
import statistics
import time
from pathlib import Path

import rospy
from spinal.msg import ESCTelemetryArray

FIELDS = ("temperature", "voltage", "current", "consumption", "rpm")
NUM_ROTORS = 4


class RotorStats:
    def __init__(self):
        self.last_fields = None
        self.update_stamps = []  # msg.stamp.to_sec() at each real update
        self.msg_count = 0
        self.crc_error_count = 0


def esc_tuple(esc_msg):
    return tuple(getattr(esc_msg, f) for f in FIELDS)


def rotor_msgs(msg):
    return [msg.esc_telemetry_1, msg.esc_telemetry_2, msg.esc_telemetry_3, msg.esc_telemetry_4]


class Analyzer:
    def __init__(self, topic):
        self.rotors = [RotorStats() for _ in range(NUM_ROTORS)]
        self.array_msg_count = 0
        self.array_recv_stamps = []
        self.sub = rospy.Subscriber(topic, ESCTelemetryArray, self.callback, queue_size=50)

    def callback(self, msg):
        now = rospy.Time.now().to_sec()
        stamp = msg.stamp.to_sec() if msg.stamp.to_sec() > 0.0 else now

        self.array_msg_count += 1
        self.array_recv_stamps.append(now)

        for i, esc in enumerate(rotor_msgs(msg)):
            r = self.rotors[i]
            r.msg_count += 1
            if esc.crc_error != 0:
                r.crc_error_count += 1

            fields = esc_tuple(esc)
            if r.last_fields is None or fields != r.last_fields:
                r.update_stamps.append(stamp)
                r.last_fields = fields


def interval_stats(stamps):
    if len(stamps) < 2:
        return None
    intervals = [b - a for a, b in zip(stamps[:-1], stamps[1:])]
    mean_dt = statistics.mean(intervals)
    return {
        "count": len(stamps),
        "mean_hz": 1.0 / mean_dt if mean_dt > 0 else float("nan"),
        "mean_ms": mean_dt * 1000.0,
        "std_ms": statistics.pstdev(intervals) * 1000.0 if len(intervals) > 1 else 0.0,
        "min_ms": min(intervals) * 1000.0,
        "max_ms": max(intervals) * 1000.0,
        "intervals": intervals,
    }


def print_summary(analyzer, duration):
    print("\n" + "=" * 72)
    print(f"esc_telem update analysis over {duration:.1f}s")
    print("=" * 72)

    array_stats = interval_stats(analyzer.array_recv_stamps)
    if array_stats:
        print(
            f"[topic]  {analyzer.array_msg_count} msgs, "
            f"publish rate {array_stats['mean_hz']:6.1f} Hz "
            f"(mean {array_stats['mean_ms']:.2f} ms, std {array_stats['std_ms']:.2f} ms)"
        )
    else:
        print(f"[topic]  {analyzer.array_msg_count} msgs (not enough data for a rate)")

    print("-" * 72)
    header = f"{'rotor':<6}{'updates':>9}{'update Hz':>12}{'mean(ms)':>11}{'std(ms)':>10}{'min(ms)':>10}{'max(ms)':>10}{'crc_err':>10}"
    print(header)

    for i, r in enumerate(analyzer.rotors):
        s = interval_stats(r.update_stamps)
        crc_pct = 100.0 * r.crc_error_count / r.msg_count if r.msg_count else 0.0
        if s:
            print(
                f"{i + 1:<6}{s['count']:>9}{s['mean_hz']:>12.2f}{s['mean_ms']:>11.2f}"
                f"{s['std_ms']:>10.2f}{s['min_ms']:>10.2f}{s['max_ms']:>10.2f}"
                f"{crc_pct:>9.2f}%"
            )
        else:
            print(f"{i + 1:<6}{len(r.update_stamps):>9}{'--':>12}{'--':>11}{'--':>10}{'--':>10}{'--':>10}{crc_pct:>9.2f}%")

    print("=" * 72)
    if array_stats:
        expected_hz = array_stats["mean_hz"] / NUM_ROTORS
        print(f"(sanity check: topic Hz / {NUM_ROTORS} rotors = {expected_hz:.2f} Hz -- "
              f"each rotor should update roughly this often if the round-robin never drops a slot)")


def save_csv(analyzer, out_dir):
    stamp_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    out_path = Path(out_dir) / f"ESCTelemUpdateAnalysis_{stamp_str}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rotor", "update_stamp_sec"])
        for i, r in enumerate(analyzer.rotors):
            for t in r.update_stamps:
                writer.writerow([i + 1, f"{t:.6f}"])
    rospy.loginfo("Saved per-rotor update timestamps to %s", out_path)


def main():
    rospy.init_node("esc_telem_update_analysis", anonymous=True)

    topic = rospy.get_param("~topic", "/beetle_omni/esc_telem")
    duration = float(rospy.get_param("~duration", 20.0))
    do_save_csv = bool(rospy.get_param("~save_csv", False))

    analyzer = Analyzer(topic)
    rospy.loginfo("Subscribed to %s, recording for %s", topic, "until Ctrl+C" if duration <= 0 else f"{duration:.1f}s")

    start = time.time()
    rate = rospy.Rate(10)
    try:
        while not rospy.is_shutdown():
            if 0 < duration <= (time.time() - start):
                break
            rate.sleep()
    except rospy.ROSInterruptException:
        pass

    elapsed = time.time() - start
    print_summary(analyzer, elapsed)

    if do_save_csv and analyzer.array_msg_count > 0:
        save_csv(analyzer, Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
