const mqtt = require("mqtt");


async function callTTS(text) {
  const encoded = encodeURIComponent(text);

  const url = `http://127.0.0.1:59531/v2/tts?text=${encoded}&voice=31&lang=ko&static=0&isPlay=1`;

  try {
    const res = await fetch(url);

    console.log("status:", res.status);

    // 응답은 그냥 소비만 (버림)
    await res.arrayBuffer();

    console.log("TTS 호출 완료");
  } catch (err) {
    console.error("에러:", err);
  }
}

// 브로커 주소 수정
//const brokerUrl = "mqtt://192.168.0.34:1883";
const brokerUrl = "mqtt://192.168.10.202:1883";
const _ID = '531e143e-fc73-4f4d-b772-27883ccb4e19'
const _IP = '192.168.0.34'
const _SEQ = '6cd69007-9a60-47e3-b7b2-a64630888371'


async function init(){
  const res = await fetch(`http://${_IP}:3000/api/v2/robot/${_ID}/load-map?pcdPath=%2Fhome%2Funitree%2Fcirculus.pcd`)
  console.log('init', res.status)
}

init()


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
      callTTS('동기화를 수행합니다')
      url = `http://${_IP}:3000/api/v2/robot/${_ID}/load-map?pcdPath=%2Fhome%2Funitree%2Fcirculus.pcd`
      break;    
    case 'start':
      callTTS('정찰을 수행합니다')
      url = `http://${_IP}:3000/api/v2/robot/${_ID}/execute-sequence/${_SEQ}?repeat=100`
      break;
    case 'end':
      callTTS('정찰을 종료합니다')
      url = `http://${_IP}:3000/api/v2/robot/${_ID}/stop-sequence`
      break;
    case 'abort':
      url = ''
      break;
    case 'stop':
      callTTS('정찰을 멈춥니다')
      cmd = 'pause' 
    case 'pause':
      callTTS('정찰을 일시정지 합니다')
    case 'resume':
      callTTS('정찰을 재개합니다')
      url = `http://${_IP}:3000/api/v2/robot/${_ID}/${cmd}`
      break;
    default:
      console.log(topic,msg)
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
