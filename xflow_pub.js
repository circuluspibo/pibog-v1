const mqtt = require("mqtt");

const brokerUrl = "mqtt://127.0.0.1:1883";

const client = mqtt.connect(brokerUrl, {
  clientId: "xms-host",
  clean: false,
  will: {
    topic: "xms/lwt",
    payload: "xmscore",
    qos: 1,
    retain: true,
  },
});

client.on("connect", () => {
  console.log("✅ Host connected");

 // const cmd = "xms/vcp/v4/HR_B1/cmd/status"


const data = {
  id : "r1",
  x: 1.5, 
  y: 1.5, 
  z: 1.5,
  sequence : 4,
  workingState :'patrol',
  soc : 0,
  errorCode : [],
  deviceHealth : '',
  custom : {
    q_x: 0.0, 
    q_y: 0.0, 
    q_z: 0.0, 
    q_w: 1.0,
    bat_pct :73.00,
    bat_vol :50871.00, 
    bat_amp : 1944.00, 
    bat_temp :27.00,
    cpu_temp :51.77,
    cpu_usage :36.13,
    cpu_mem :33.87,
    cpu_freq :960.00,
    motor_temp_max: 50.00,
    motor_temp_avg :39.21,
    motor_err_cnt : 0,
    motor_temps:[32.0,32.0,32.0,32.0,37.0,39.0,32.0,33.0,32.0,32.0,40.0,40.0,33.0,39.0,42.0,44.0,45.0,44.0,42.0,45.0,40.0,41.0,50.0,45.0,45.0,45.0,43.0,40.0,41.0],
    motor_errors :[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
  }, 
}

const cmd = "xflow/rcp/v1/r1/cmd/status"

  // connect 되었음을 알림 (null 메시지)
  client.publish(cmd, JSON.stringify(data), {
    qos: 1,
    retain: true,
  });

  console.log(`📤 Published ${cmd} to xms/lwt`);
});

client.on("error", (err) => {
  console.error("⚠️ Error:", err);
});