const mqtt = require("mqtt");

// 브로커 주소 수정
const brokerUrl = "mqtt://127.0.0.1:1883";
const _ID = '5480f373-e20c-4db9-90cb-9d49f56aac70'
const _IP = '192.168.21.19'
const _SEQ = '669cd346-2b89-4eda-93bc-4e833d711d63'

const client = mqtt.connect(brokerUrl, {
  clientId: "monitor-client",
  clean: true,
});

client.on("connect", () => {
  console.log("✅ Connected to broker");
  client.subscribe("xflow/rcp/v1/H1/cmd/+", (err) => {
    if (err) {
      console.error("❌ Subscribe error:", err);
    } else {
      console.log("📡 Subscribed to xflow");
    }
  });
});

client.on("message", async (topic, message) => {
  const msg = message.toString() || "(null)";
  console.log(`📩 ${topic} -> ${msg}`);

  const cmd = topic.replace('xflow/rcp/v1/H1/cmd/','')
  console.log('cmd',cmd)
  let url = ''

  switch(cmd){
    case 'sync':
      url = `http://${_IP}:3000/api/v2/robot/${_ID}/load-map?pcdPath=%2Fhome%2Funitree%2Fcirculus.pcd`
      break;    
    case 'start':
      url = `http://${_IP}:3000/api/v2/robot/${_ID}/execute-sequence/${_SEQ}?repeat=3`
      break;
    case 'end':
      url = `http://${_IP}:3000/api/v2/robot/${_ID}/stop-sequence`
      break;
    case 'abort':
      url = ''
      break;
    case 'stop':
    case 'pause':
    case 'resume':
      url = `http://${_IP}:3000/api/v2/robot/${_ID}/${cmd}`
      break;
    default:
      console.log('not allowed',cmd)
  }

  try {
    // 1. fetch로 데이터 요청
    console.log('request',url)
    const response = await fetch(url);
    
    // 2. 응답이 정상인지 확인
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    // 3. JSON 데이터 파싱
    const data = await response.json();
    console.log('Return',data);
  } catch (error) {
    console.error('Fetch error:', error);
  }


  


});

client.on("close", () => {
  console.log("🔌 Connection closed");
});

client.on("error", (err) => {
  console.error("⚠️ Error:", err);
});