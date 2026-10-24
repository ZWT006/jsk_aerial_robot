/*
 * can_core.cpp
 *
 *  Created on: 2016/10/24
 *      Author: anzai
 */

#include "can_core.h"

#ifdef STM32F7
namespace CAN {

  namespace {
    //CAN bus registers
    CAN_HandleTypeDef* hcan_;
    volatile uint32_t tx_drop_count = 0;

    void recordTxDrop()
    {
      uint32_t primask = __get_PRIMASK();
      __disable_irq();
      tx_drop_count++;
      if (primask == 0U) __enable_irq();
    }
  }

  void init(CAN_HandleTypeDef* hcan)
  {
    hcan_ = hcan;

    CAN_FilterTypeDef sFilterConfig;
    sFilterConfig.FilterBank = 0;
    sFilterConfig.FilterMode = CAN_FILTERMODE_IDMASK;
    sFilterConfig.FilterScale = CAN_FILTERSCALE_32BIT;
    sFilterConfig.FilterIdHigh = 0x0000;
    sFilterConfig.FilterIdLow = 0x0000;
    sFilterConfig.FilterMaskIdHigh = 0x0000;
    sFilterConfig.FilterMaskIdLow = 0x0000;
    sFilterConfig.FilterFIFOAssignment = CAN_FILTER_FIFO1;
    sFilterConfig.SlaveStartFilterBank = 14;
    sFilterConfig.FilterActivation = ENABLE;
    HAL_CAN_ConfigFilter(hcan_, &sFilterConfig);

  }

  CAN_HandleTypeDef* getHcanInstance()
  {
    return hcan_;
  }

  uint32_t getTxDropCount()
  {
    return tx_drop_count;
  }

  bool sendMessage(uint8_t device_id, uint8_t message_id, uint8_t slave_id, uint32_t dlc, uint8_t* data, uint32_t timeout)
  {
    if (dlc > 8 || (dlc > 0 && data == nullptr)) {
      recordTxDrop();
      return false;
    }

    CAN_TxHeaderTypeDef tx_header = {};
    uint32_t tx_mailbox = 0;
    uint8_t tx_data[8] = {};
    tx_header.RTR = CAN_RTR_DATA;
    tx_header.IDE = CAN_ID_STD;
    tx_header.StdId = (((device_id & ((1 << DEVICE_ID_LEN) - 1))  << (MESSAGE_ID_LEN + SLAVE_ID_LEN))) | ((message_id & ((1 << MESSAGE_ID_LEN) - 1)) << SLAVE_ID_LEN) | (slave_id & ((1 << SLAVE_ID_LEN) - 1));
    tx_header.DLC = dlc;
    if (dlc > 0) memcpy(tx_data, data, sizeof(uint8_t) * dlc);

    uint32_t tickstart = HAL_GetTick();
    while (1) {
      HAL_StatusTypeDef status = HAL_BUSY;
      uint32_t primask = __get_PRIMASK();
      __disable_irq();
      if (HAL_CAN_GetTxMailboxesFreeLevel(hcan_) > 0) {
        status = HAL_CAN_AddTxMessage(hcan_, &tx_header, tx_data, &tx_mailbox);
      }
      if (primask == 0U) __enable_irq();
      if (status == HAL_OK) return true;

      if ((timeout == 0U) || ((HAL_GetTick() - tickstart) > timeout)) {
        recordTxDrop();
        return false;
      }
    }
  }
}
#endif

#ifdef STM32H7
namespace CAN {

  namespace {
    //CAN bus registers
    FDCAN_HandleTypeDef* hfdcan_;
    volatile uint32_t tx_drop_count = 0;
    ErrorStats error_stats = {};

    constexpr uint8_t PROTOCOL_ERROR_NONE = 0;
    constexpr uint8_t PROTOCOL_ERROR_NO_CHANGE = 7;

    void recordTxDrop()
    {
      uint32_t primask = __get_PRIMASK();
      __disable_irq();
      tx_drop_count++;
      if (primask == 0U) __enable_irq();
    }
  }

  void updateErrorStatus()
  {
    if (hfdcan_ == nullptr) return;
    FDCAN_GlobalTypeDef* instance = hfdcan_->Instance;

    // Reading ECR clears CEL and reading PSR resets LEC/DLEC, so this must
    // remain the only reader of both registers.
    const uint32_t ecr = instance->ECR;
    const uint32_t psr = instance->PSR;

    error_stats.error_count += (ecr & FDCAN_ECR_CEL) >> FDCAN_ECR_CEL_Pos;
    const uint8_t tec = (ecr & FDCAN_ECR_TEC) >> FDCAN_ECR_TEC_Pos;
    const uint8_t rec = (ecr & FDCAN_ECR_REC) >> FDCAN_ECR_REC_Pos;
    if (tec > error_stats.tec_max) error_stats.tec_max = tec;
    if (rec > error_stats.rec_max) error_stats.rec_max = rec;

    const uint8_t lec = (psr & FDCAN_PSR_LEC) >> FDCAN_PSR_LEC_Pos;
    if (lec != PROTOCOL_ERROR_NONE && lec != PROTOCOL_ERROR_NO_CHANGE) {
      error_stats.last_error_code = lec;
    }
    const uint8_t dlec = (psr & FDCAN_PSR_DLEC) >> FDCAN_PSR_DLEC_Pos;
    if (dlec != PROTOCOL_ERROR_NONE && dlec != PROTOCOL_ERROR_NO_CHANGE) {
      error_stats.last_data_error_code = dlec;
    }

    // On bus-off the controller sets CCCR.INIT and stays off the bus until
    // software clears it; it then rejoins after 129 x 11 recessive bits.
    if ((psr & FDCAN_PSR_BO) != 0U &&
        (instance->CCCR & FDCAN_CCCR_INIT) != 0U) {
      error_stats.bus_off_count++;
      error_stats.tec_max = 0xFF;
      CLEAR_BIT(instance->CCCR, FDCAN_CCCR_INIT);
    }
  }

  ErrorStats takeErrorStats()
  {
    const ErrorStats stats = error_stats;
    error_stats.tec_max = 0;
    error_stats.rec_max = 0;
    return stats;
  }

  void init(FDCAN_HandleTypeDef* hfdcan)
  {
    hfdcan_ = hfdcan;

    FDCAN_FilterTypeDef sFilterConfig;
    sFilterConfig.IdType = FDCAN_STANDARD_ID;
    sFilterConfig.FilterIndex = 0;
    sFilterConfig.FilterType = FDCAN_FILTER_MASK;
    sFilterConfig.FilterConfig = FDCAN_FILTER_TO_RXFIFO0;
    sFilterConfig.FilterID1 = 0x000;
    sFilterConfig.FilterID2 = 0x000;
    HAL_FDCAN_ConfigFilter(hfdcan_, &sFilterConfig);
    /*
       Configure global filter:
       - Reject all remote frames with STD and EXT ID
       - Reject non matching frames with STD ID and EXT ID
    */
    HAL_FDCAN_ConfigGlobalFilter(hfdcan_, FDCAN_REJECT, FDCAN_REJECT, FDCAN_REJECT_REMOTE, FDCAN_REJECT_REMOTE);

    // www.programmersought.com/article/43428259685/#TDC_568
    // The SSP location is defined as the sum of the measurement delay from the FDCAN_TX
    // pin to the FDCAN_RX pin, plus the transmitter delay compensation offset configured
    // by the TDCO [6: 0] field.
    // TdcOffset = DataTimeSeg1 * DataPrescaler.
    HAL_FDCAN_ConfigTxDelayCompensation(hfdcan_, 7, 0); //Configure hfdcan1
    HAL_FDCAN_EnableTxDelayCompensation(hfdcan_); //Enable TDC of hfdcan1
  }

  FDCAN_HandleTypeDef* getHcanInstance()
  {
    return hfdcan_;
  }

  uint32_t getTxDropCount()
  {
    return tx_drop_count;
  }

  bool sendMessage(uint8_t device_id, uint8_t message_id, uint8_t slave_id, uint32_t dlc, uint8_t* data, uint32_t timeout)
  {
    if (dlc > 64 || (dlc > 0 && data == nullptr)) {
      recordTxDrop();
      return false;
    }

    FDCAN_TxHeaderTypeDef tx_header = {};
    uint8_t tx_data[64] = {};
    tx_header.Identifier = (((device_id & ((1 << DEVICE_ID_LEN) - 1))  << (MESSAGE_ID_LEN + SLAVE_ID_LEN))) | ((message_id & ((1 << MESSAGE_ID_LEN) - 1)) << SLAVE_ID_LEN) | (slave_id & ((1 << SLAVE_ID_LEN) - 1));
    tx_header.IdType = FDCAN_STANDARD_ID;
    tx_header.TxFrameType = FDCAN_DATA_FRAME;
    tx_header.ErrorStateIndicator = FDCAN_ESI_ACTIVE;
    tx_header.TxEventFifoControl = FDCAN_NO_TX_EVENTS;
    tx_header.MessageMarker = 0;

    if (dlc <= 8) {
      tx_header.FDFormat = FDCAN_CLASSIC_CAN;
      tx_header.BitRateSwitch = FDCAN_BRS_OFF;
      tx_header.DataLength = dlc << 16;
    } else {
      tx_header.FDFormat = FDCAN_FD_CAN;
      tx_header.BitRateSwitch = FDCAN_BRS_ON;
      switch(dlc) {
      case 12:
        {
          tx_header.DataLength = FDCAN_DLC_BYTES_12;
          break;
        }
      case 16:
        {
          tx_header.DataLength = FDCAN_DLC_BYTES_16;
          break;
        }
      case 20:
        {
          tx_header.DataLength = FDCAN_DLC_BYTES_20;
          break;
        }
      case 24:
        {
          tx_header.DataLength = FDCAN_DLC_BYTES_24;
          break;
        }
      case 32:
        {
          tx_header.DataLength = FDCAN_DLC_BYTES_32;
          break;
        }
      case 48:
        {
          tx_header.DataLength = FDCAN_DLC_BYTES_48;
          break;
        }
      case 64:
        {
          tx_header.DataLength = FDCAN_DLC_BYTES_64;
          break;
        }
      default:
        {
          recordTxDrop();
          return false;
        }
      }
    }

    if (dlc > 0) memcpy(tx_data, data, sizeof(uint8_t) * dlc);

    uint32_t tickstart = HAL_GetTick();
    while (1) {
      uint32_t primask = __get_PRIMASK();
      __disable_irq();
      HAL_StatusTypeDef status = HAL_FDCAN_AddMessageToTxFifoQ(
        getHcanInstance(), &tx_header, tx_data);
      if (primask == 0U) __enable_irq();
      if (status == HAL_OK) return true;

      if ((timeout == 0U) || ((HAL_GetTick() - tickstart) > timeout)) {
        recordTxDrop();
        return false;
      }
    }
  }
}
#endif
