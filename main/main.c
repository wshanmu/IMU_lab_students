#include "bmi270.h"
#include "driver/gpio.h"
#include "driver/i2c_master.h"
#include "esp_err.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define I2C_MASTER_NUM I2C_NUM_0
#define I2C_MASTER_SDA_IO // TODO: Set the GPIO number for I2C SDA to communicate with the BMI270 sensor
#define I2C_MASTER_SCL_IO // TODO: Set the GPIO number for I2C SCL to communicate with the BMI270 sensor

static const char *TAG = "BMI270";

static esp_err_t bmi270_init(i2c_master_bus_handle_t i2c_bus, bmi270_handle_t **bmi270)
{
    const uint8_t addresses[] = {BMI270_I2C_ADDRESS_L, BMI270_I2C_ADDRESS_H};
    esp_err_t last_error = ESP_ERR_NOT_FOUND;

    for (size_t i = 0; i < sizeof(addresses) / sizeof(addresses[0]); ++i) {
        const bmi270_driver_config_t driver_config = {
            .addr = addresses[i],
            .interface = BMI270_USE_I2C,
            .i2c_bus = i2c_bus,
        };

        last_error = bmi270_create(&driver_config, bmi270);
        if (last_error == ESP_OK) {
            ESP_LOGI(TAG, "Found BMI270 at I2C address 0x%02x", addresses[i]);
            return ESP_OK;
        }
    }

    return last_error;
}

void app_main(void)
{
    i2c_master_bus_handle_t i2c_bus = NULL;
    bmi270_handle_t *bmi270 = NULL;

    const i2c_master_bus_config_t i2c_bus_config = {
        .i2c_port = I2C_MASTER_NUM,
        .sda_io_num = I2C_MASTER_SDA_IO,
        .scl_io_num = I2C_MASTER_SCL_IO,
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .glitch_ignore_cnt = 7,
        .flags.enable_internal_pullup = true,
    };
    ESP_ERROR_CHECK(i2c_new_master_bus(&i2c_bus_config, &i2c_bus));

    ESP_ERROR_CHECK(bmi270_init(i2c_bus, &bmi270));

    const bmi270_config_t bmi270_config = {
        .acce_odr = BMI270_ACC_ODR_100_HZ,
        .acce_range = BMI270_ACC_RANGE_4_G,
        .gyro_odr = BMI270_GYR_ODR_100_HZ,
        .gyro_range = BMI270_GYR_RANGE_1000_DPS,
    };
    ESP_ERROR_CHECK(bmi270_start(bmi270, &bmi270_config));

    while (true) {
        float ax, ay, az;
        float gx, gy, gz;

        ESP_ERROR_CHECK(bmi270_get_acce_data(bmi270, &ax, &ay, &az));
        ESP_ERROR_CHECK(bmi270_get_gyro_data(bmi270, &gx, &gy, &gz));

        ESP_LOGI(TAG,
                 "accel[g] x=% .3f y=% .3f z=% .3f | gyro[dps] x=% .2f y=% .2f z=% .2f",
                 ax, ay, az, gx, gy, gz);

        vTaskDelay(pdMS_TO_TICKS(SOME_DELAY_MS)); // TODO: Set the delay time between readings so that the FPS will be 100Hz. The delay time should be in milliseconds.
    }
}
