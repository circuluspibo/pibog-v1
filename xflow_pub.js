const mqtt = require("mqtt");
const readline = require("readline");

// 1. MQTT 및 설정 값
const brokerUrl = "mqtt://127.0.0.1:1883";
const robotId = "r1";
let sequence = 1;

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
    payload = { id: robotId, workingState: "patrol", custom: { bat_pct: 73.0 } }; 
  }

  client.publish(topic, JSON.stringify(payload), { qos: 1 }, (err) => {
    if (!err) {
      console.log(`\n📤 [PUBLISH] Topic: ${topic}`);
      console.log(`📦 Payload: ${JSON.stringify(payload)}`);
    }
  });
}

client.on("error", (err) => console.error("⚠️ MQTT Error:", err));
