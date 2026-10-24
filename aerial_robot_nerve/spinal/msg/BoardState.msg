uint8 slave_id
uint8 sequence
uint32 uptime_ms
uint32 state_age_ms
uint16 servo_target_count
uint16 thrust_target_count
uint16 servo_pvc_count
uint16 rotor_esc_status_count
uint32 can_rx_drop_count
uint32 can_tx_drop_count
# FDCAN protocol error statistics; tec/rec max are over the last report interval
uint32 can_error_count
uint16 can_bus_off_count
uint8 can_tec_max
uint8 can_rec_max
uint8 can_last_error_code
uint8 can_last_data_error_code
int16[] servo_control_count
