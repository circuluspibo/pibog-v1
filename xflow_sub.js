const mqtt = require("mqtt");

// 브로커 주소 수정
const brokerUrl = "mqtt://127.0.0.1:1883";

const client = mqtt.connect(brokerUrl, {
  clientId: "monitor-client",
  clean: true,
});

client.on("connect", () => {
  console.log("✅ Connected to broker");
  client.subscribe("xflow/rcp/v1/r1/cmd/+", (err) => {
    if (err) {
      console.error("❌ Subscribe error:", err);
    } else {
      console.log("📡 Subscribed to xflow");
    }
  });
});

client.on("message", (topic, message) => {
  const msg = message.toString() || "(null)";
  console.log(`📩 ${topic} -> ${msg}`);
});

client.on("close", () => {
  console.log("🔌 Connection closed");
});

client.on("error", (err) => {
  console.error("⚠️ Error:", err);
});