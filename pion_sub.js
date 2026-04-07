const robotId = "H1"; // 예시 ID

// 쿼터니언(Quaternion)을 방향 벡터(Heading Vector)로 변환하는 함수
// 여기서는 '앞방향(Forward)'을 기준으로 계산합니다.
function getHeadingFromQuaternion(qx, qy, qz, qw) {
  // 로봇의 로컬 축이 Z축이 앞방향이라고 가정할 때의 변환식
  const x = 2 * (qx * qz + qw * qy);
  const y = 2 * (qy * qz - qw * qx);
  const z = 1 - 2 * (qx * qx + qy * qy);
  return [parseFloat(x.toFixed(4)), parseFloat(y.toFixed(4)), parseFloat(z.toFixed(4))];
}




/*

const robotData = {"batteryPower":36,"batteryVol":46965,"batteryAmp":-1877,"batteryTemp":31,"cpuUsage":44.61,"cpuMemory":37.62,"cpuTemp":54.54,"cpuFrequency":1356,"motorTempMax":52,"motorTempAvg":39.66,"motorErrCnt":0,"motorTemp":[32,32,32,33,40,41,33,34,33,33,43,42,34,40,44,46,46,45,42,42,33,33,52,47,48,47,44,39,40],"motorError":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"_ts":1775437932370};
const positionData = {"x":0.0306,"y":-0.0135,"z":-0.0479,"q_x":-0.003,"q_y":-0.0641,"q_z":0.0489,"q_w":0.9967,"_ts":1775437932370};

const robotId = "pion-01"; // 예시 ID

// 쿼터니언(Quaternion)을 방향 벡터(Heading Vector)로 변환하는 함수
// 여기서는 '앞방향(Forward)'을 기준으로 계산합니다.
function getHeadingFromQuaternion(qx, qy, qz, qw) {
  // 로봇의 로컬 축이 Z축이 앞방향이라고 가정할 때의 변환식
  const x = 2 * (qx * qz + qw * qy);
  const y = 2 * (qy * qz - qw * qx);
  const z = 1 - 2 * (qx * qx + qy * qy);
  return [parseFloat(x.toFixed(4)), parseFloat(y.toFixed(4)), parseFloat(z.toFixed(4))];
}

const statusData = {
  id: robotId,
  // 위치 데이터 매핑
  x: positionData.x,
  y: positionData.y,
  z: positionData.z,
  
  // 시퀀스는 타임스탬프 또는 별도 카운터 사용
  sequence: positionData._ts,
  
  // 상태 추론 (데이터에 명시되지 않은 경우 조건부 설정)
  workingState: robotData.motorAmp !== 0 ? "R" : "I", // 전류 흐르면 Running, 아니면 Idle
  taskType: "Patrol", // 시나리오에 따라 고정 또는 로직 처리
  
  // 이벤트 및 모션 (기본값 설정)
  detectionEvent: robotData.motorErrCnt > 0 ? "ObstacleDetected" : "", 
  motionState: "T", // Moving
  
  // 쿼터니언 기반 헤딩 계산
  heading: getHeadingFromQuaternion(
    positionData.q_x, 
    positionData.q_y, 
    positionData.q_z, 
    positionData.q_w
  ),
  
  // 배터리 및 에러 정보
  soc: robotData.batteryPower, // % 단위
  errorCode: robotData.motorError.filter(code => code !== 0), // 0이 아닌 에러만 필터링
  
  // 장치 헬스 체크
  deviceHealth: robotData.motorErrCnt === 0 && robotData.cpuTemp < 80 ? "Ok" : "Fault",
  
  // 원본 데이터 포함
  custom: {
    ...robotData,
    ...positionData
  },
};

console.log(statusData);

*/

const mqtt = require("mqtt");

// 브로커 주소 수정
const brokerUrl = "mqtt://192.168.0.34:1883";

const client = mqtt.connect(brokerUrl, {
  clientId: "monitor-client",
  clean: true,
});


let robotData = 0
let positionData = 0

client.on("connect", () => {
  console.log("✅ Connected to broker");
  client.subscribe("arcos/robot/+/position", (err) => {
    if (err) {
      console.error("❌ Subscribe error:", err);
    } else {
      console.log("📡 Subscribed to arcos");
    }
  });
  client.subscribe("arcos/robot/+/robot_data", (err) => {
    if (err) {
      console.error("❌ Subscribe error:", err);
    } else {
      console.log("📡 Subscribed to arcos");
    }
  });  
});

client.on("message", async (topic, message) => {
  const msg = message.toString() || "(null)";
  console.log(`📩 ${topic} -> ${msg}`);

  if(topic.endsWith('robot_data'))
    robotData = JSON.parse(msg)
  else if(topic.endsWith('position'))
    positionData = JSON.parse(msg)
});

if(robotData && positionData){
  const statusData = {
    id: robotId,
    // 위치 데이터 매핑
    x: positionData.x,
    y: positionData.y,
    z: positionData.z,
    
    // 시퀀스는 타임스탬프 또는 별도 카운터 사용
    sequence: positionData._ts,
    
    // 상태 추론 (데이터에 명시되지 않은 경우 조건부 설정)
    workingState: robotData.motorAmp !== 0 ? "R" : "I", // 전류 흐르면 Running, 아니면 Idle
    taskType: "Patrol", // 시나리오에 따라 고정 또는 로직 처리
    
    // 이벤트 및 모션 (기본값 설정)
    detectionEvent: robotData.motorErrCnt > 0 ? "ObstacleDetected" : "", 
    motionState: "T", // Moving
    
    // 쿼터니언 기반 헤딩 계산
    heading: getHeadingFromQuaternion(
      positionData.q_x, 
      positionData.q_y, 
      positionData.q_z, 
      positionData.q_w
    ),
    
    // 배터리 및 에러 정보
    soc: robotData.batteryPower, // % 단위
    errorCode: robotData.motorError.filter(code => code !== 0), // 0이 아닌 에러만 필터링
    
    // 장치 헬스 체크
    deviceHealth: robotData.motorErrCnt === 0 && robotData.cpuTemp < 80 ? "Ok" : "Fault",
    
    // 원본 데이터 포함
    custom: {
      ...robotData,
      ...positionData
    },
  };
  /* 이거는 HCP 에 접속되어야 함.. 로봇과 다름
  client.publish(`xflow/hcp/v1/${robotId}/cmd/${command}`, JSON.stringify(payload), { qos: 1 }, (err) => {
    if (!err) {
      console.log(`\n📤 [PUBLISH] hcp Topic: ${topic}`);
      console.log(`📦 Payload: ${JSON.stringify(payload)}`);
    }
  })
  */
}

client.on("close", () => {
  console.log("🔌 Connection closed");
});

client.on("error", (err) => {
  console.error("⚠️ Error:", err);
});