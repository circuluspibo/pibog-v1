const mqtt = require("mqtt");

// 브로커 주소 수정
//const brokerUrl = "mqtt://192.168.0.34:1883";
const brokerUrl = "mqtt://192.168.10.202:1883";
const _ID = '531e143e-fc73-4f4d-b772-27883ccb4e19'
const _IP = '192.168.0.34'
const _SEQ = '6cd69007-9a60-47e3-b7b2-a64630888371'

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
      return
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