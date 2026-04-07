const mqtt = require("mqtt");
const readline = require("readline");

// 1. MQTT 및 설정 값
//const brokerUrl = "mqtt://192.168.0.34:1883";
const brokerUrl = "mqtt://192.168.10.202:1883";
const robotId = "H1";
let sequence = 1;

/*
📩 arcos/robot/pion/robot_data -> {"batteryPower":36,"batteryVol":46965,"batteryAmp":-1877,"batteryTemp":31,"cpuUsage":44.61,"cpuMemory":37.62,"cpuTemp":54.54,"cpuFrequency":1356,"motorTempMax":52,"motorTempAvg":39.66,"motorErrCnt":0,"motorTemp":[32,32,32,33,40,41,33,34,33,33,43,42,34,40,44,46,46,45,42,42,33,33,52,47,48,47,44,39,40],"motorError":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"_ts":1775437932370}
📩 arcos/robot/pion/position -> {"x":0.0306,"y":-0.0135,"z":-0.0479,"q_x":-0.003,"q_y":-0.0641,"q_z":0.0489,"q_w":0.9967,"_ts":1775437932370}

*/


const statusData = {
  id: robotId,
  x: 1.5, y: 1.5, z: 1.5,
  sequence: 4,
  workingState : "R", // I, C, P
  taskType: "Patrol", // Return // Null
  detectionEvent : "", //ObstacleDetected PersonWithHelmet PersonWithoutHelmet
  motionState : "", // T C X A
  heading : "",
  soc: 0,
  errorCode: [],
  deviceHealth: "Ok", // fault
  custom: {
    q_x: 0.0, q_y: 0.0, q_z: 0.0, q_w: 1.0,
    bat_pct: 73.0, bat_vol: 50871.0, bat_amp: 1944.0, bat_temp: 27.0,
    cpu_temp: 51.77, cpu_usage: 36.13, cpu_mem: 33.87, cpu_freq: 960.0,
    motor_temp_max: 50.0, motor_temp_avg: 39.21, motor_err_cnt: 0,
    motor_temps: [32.0, 32.0, 32.0, 32.0, 37.0, 39.0, 32.0, 33.0, 32.0, 32.0, 40.0, 40.0, 33.0, 39.0, 42.0, 44.0, 45.0, 44.0, 42.0, 45.0, 40.0, 41.0, 50.0, 45.0, 45.0, 45.0, 43.0, 40.0, 41.0],
    motor_errors: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  },
};

const client = mqtt.connect(brokerUrl, {
  clientId: "xms-host-manager",
  clean: false,
});

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

// 2. 명령어 메뉴 정의
const COMMAND_INFO = {
  sync : "작업 준비",
  start: "작업 시작 (jobId: 1, recipeId: 1)",
  stop: "작업 중지",
  end: "작업 종료",
  abort: "즉시 중단 (비상)",
  pause: "일시 정지",
  resume: "다시 시작",
  status: "현재 로봇 상태 보고 (Data 전송)",
  help: "명령어 목록 다시 보기",
  exit: "프로그램 종료",
};

function showMenu() {
  console.log("\n" + "=".repeat(50));
  console.log(`🤖 [Robot: ${robotId}] 제어 터미널`);
  console.log("-".repeat(50));
  Object.entries(COMMAND_INFO).forEach(([cmd, desc]) => {
    // 명령어는 왼쪽 정렬, 설명은 그 뒤에 붙임
    console.log(` > ${cmd.padEnd(10)} : ${desc}`);
  });
  console.log("=".repeat(50));
}

// 3. 로직 처리
client.on("connect", () => {
  console.log("✅ MQTT Broker 연결 성공");
  showMenu();
  askCommand();
});

function askCommand() {
  rl.question("\n명령어를 입력하세요 >> ", (input) => {
    const command = input.trim().toLowerCase();

    if (command === "exit") {
      console.log("👋 프로그램을 종료합니다.");
      process.exit(0);
    }

    if (command === "help") {
      showMenu();
      return askCommand();
    }

    if (COMMAND_INFO[command]) {
      publishCommand(command);
    } else {
      console.log(`❌ [${command}]는 알 수 없는 명령어입니다. 'help'를 입력해 보세요.`);
    }

    askCommand(); // 다시 입력 대기
  });
}

function publishCommand(command) {
  const topic = `xflow/rcp/v1/${robotId}/cmd/${command}`;
  let payload = { refSeq: sequence++ };

  // 특정 명령어에 대한 추가 파라미터 처리
  if (command === "start") {
    payload.jobId = "1";
    payload.recipeId = "1";
  } else if (command === "status") {
    // 기존에 작성하셨던 복잡한 상태 데이터를 여기에 할당
    payload = { id: robotId, workingState: "patrol", custom: statusData }; 
  }
  if (command === "status") {
    const st = `xflow/hcp/v1/${robotId}/cmd/${command}`
    client.publish(`xflow/hcp/v1/${robotId}/cmd/${command}`, JSON.stringify(payload), { qos: 1 }, (err) => {
      if (!err) {
        console.log(`\n📤 [PUBLISH] hcp Topic: ${st}`);
        console.log(`📦 Payload: ${JSON.stringify(payload)}`);
      }
    })
  } else {
    client.publish(topic, JSON.stringify(payload), { qos: 1 }, (err) => {
      if (!err) {
        console.log(`\n📤 [PUBLISH] rcp Topic: ${topic}`);
        console.log(`📦 Payload: ${JSON.stringify(payload)}`);
      }
    })   
  } 
}

client.on("error", (err) => console.error("⚠️ MQTT Error:", err))
