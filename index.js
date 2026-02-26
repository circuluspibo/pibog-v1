// 1. CommonJS require() 대신 ES Module import 사용
import fastify from 'fastify';
import fastifyStatic from '@fastify/static';
import fastifyCors from '@fastify/cors';
import fastifyWebsocket from '@fastify/websocket';
import fastifySwagger from '@fastify/swagger';
import fastifySwaggerUi from '@fastify/swagger-ui';
import { SerialPort } from 'serialport';
import fs from 'fs/promises';
import path from 'path';
import crypto from 'crypto';
import { fileURLToPath } from 'url';
import os from 'os';
import { spawn } from "child_process";
import http from "http";

const TTS_URL = 'http://127.0.0.1:59530/v1/tts?text="';

async function fetchAndPlay(text) {
  try {
    console.log("TTS 요청 중...");

    const response = await fetch(`${TTS_URL}${text}`);

    if (!response.ok) {
      throw new Error(`HTTP error: ${response.status}`);
    }

    // play 프로세스 실행 (stdin으로 받기)
    const player = spawn("play", ["-"]);

    // Web Stream → Node Stream 변환
    const readable = response.body;

    readable.pipeTo(
      new WritableStream({
        write(chunk) {
          player.stdin.write(chunk);
        },
        close() {
          player.stdin.end();
        }
      })
    );

    player.on("close", (code) => {
      console.log("재생 종료:", code);
    });

  } catch (err) {
    console.error("에러:", err.message);
  }
}

// __dirname 대체 (ESM 환경에서는 __dirname이 정의되지 않음)
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = 8000

const app = fastify({ logger: true });

// 전역 변수
let ser = null;
let state = {};
const SERIAL_PORT = '/dev/ttyACM0';
const BAUDRATE = 115200;
const MOTION_DATA_PATH = path.join(__dirname, 'motion_data.json');

// 유틸리티 함수
function getHash(text) {
  return crypto.createHash('md5').update(text, 'utf-8').digest('hex');
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// 시리얼 포트 초기화
async function initSerial() {
  try {
    if (ser === null || !ser.isOpen) {
      ser = new SerialPort({
        path: SERIAL_PORT,
        baudRate: BAUDRATE,
        autoOpen: false
      });
      
      await new Promise((resolve, reject) => {
        ser.open((err) => {
          if (err) reject(err);
          else resolve();
        });
      });
      
      await sleep(2000); // 포트 안정화
    }
    return true;
  } catch (e) {
    console.error('시리얼 포트 초기화 실패:', e);
    return false;
  }
}

// 모션 데이터를 바이트 명령으로 변환
function createCommand(motion) {
  return Buffer.from([
    motion.arm,
    motion.shoulder_front,
    motion.shoulder_side,
    motion.elbow_front,
    motion.elbow_side,
    motion.finger
  ]);
}

// 머리용 명령 생성
function createHeadCommand(motion) {
  return Buffer.from([
    252,
    motion.head_tilt,
    motion.head_pan,
    0,
    0,
    0
  ]);
}

// 모션 실행 유틸리티
async function move(data, delay = 1000) {
  if (ser && ser.isOpen) {
    ser.write(data);
    await sleep(delay);
  }
}

// CORS 설정
app.register(fastifyCors, {
  origin: '*',
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: '*'
});

// Swagger 설정
app.register(fastifySwagger, {
  openapi: {
    info: {
      title: 'Robot Motion Control API',
      description: 'API for controlling robot arm motions and gestures via serial communication',
      version: '1.0.0'
    },
    servers: [
      {
        url: 'http://localhost:3000',
        description: 'Development server'
      }
    ],
    tags: [
      { name: 'Motion', description: 'Single motion control endpoints' },
      { name: 'Sequence', description: 'Motion sequence management' },
      { name: 'Gestures', description: 'Pre-defined gesture endpoints' },
      { name: 'System', description: 'System and health check endpoints' },
      { name: 'WebSocket', description: 'Real-time control via WebSocket' }
    ]
  }
});

app.register(fastifySwaggerUi, {
  routePrefix: '/docs',
  uiConfig: {
    docExpansion: 'list',
    deepLinking: false
  },
  staticCSP: true,
  transformStaticCSP: (header) => header
});

// 정적 파일 서빙
app.register(fastifyStatic, {
  root: path.join(__dirname, 'web'),
  prefix: '/web/'
});

app.register(fastifyStatic, {
  root: path.join(__dirname, 'webfonts'),
  prefix: '/webfonts/',
  decorateReply: false
});

// WebSocket 지원
app.register(fastifyWebsocket);

// 서버 시작시 초기화
app.addHook('onReady', async () => {
  await initSerial();
  
  // 모션 데이터 파일이 없으면 생성
  try {
    await fs.access(MOTION_DATA_PATH);
  } catch {
    await fs.writeFile(MOTION_DATA_PATH, JSON.stringify({}));
  }
});

// 서버 종료시 시리얼 포트 닫기
app.addHook('onClose', async () => {
  if (ser && ser.isOpen) {
    ser.close();
  }
});

// 라우트 정의
app.get('/', {
  schema: {
    tags: ['System'],
    description: 'API 서버 상태 확인',
    response: {
      200: {
        type: 'object',
        properties: {
          message: { type: 'string' },
          status: { type: 'string' }
        }
      }
    }
  }
}, async (request, reply) => {
  return { message: 'Robot Motion Control API', status: 'running' };
});

// Swagger 스키마 정의 (생략 없이 그대로 유지)
const RobotMotionSchema = {
  type: 'object',  // required 'head_tilt' , 'head_pan'
  required: ['arm', 'shoulder_front', 'shoulder_side', 'elbow_front', 'elbow_side', 'finger', 'duration'],
  properties: {
    arm: {
      type: 'integer',
      description: '0xFD (253): 왼팔, 0xFE (254): 오른팔, 0xFF (255): 양팔, 0xFC (252): 머리',
      enum: [252, 253, 254, 255]
    },
    head_tilt: {
      type: 'integer',
      minimum: 0,
      maximum: 40,
      description: '머리 기울기 (0-40도)'
    },
    head_pan: {
      type: 'integer',
      minimum: 10,
      maximum: 170,
      description: '머리 회전 (10-170도)'
    },
    shoulder_front: {
      type: 'integer',
      minimum: 0,
      maximum: 160,
      description: '어깨 관절 전방 구동 각도 (0-160도)'
    },
    shoulder_side: {
      type: 'integer',
      minimum: 0,
      maximum: 100,
      description: '어깨 관절 측방 구동 각도 (0-100도)'
    },
    elbow_front: {
      type: 'integer',
      minimum: 0,
      maximum: 115,
      description: '팔꿈치 관절 전방 구동 각도 (0-115도)'
    },
    elbow_side: {
      type: 'integer',
      minimum: 0,
      maximum: 160,
      description: '팔꿈치 관절 측방 구동 각도 (0-160도, 80이 중앙)'
    },
    finger: {
      type: 'integer',
      minimum: 0,
      maximum: 63,
      description: '손가락 구동 (0:대기, 1:엄지잡기, 2:엄지놓기, 3:4손가락잡기, 6:4손가락펴기)'
    },
    duration: {
      type: 'number',
      minimum: 0,
      description: '모션 지속 시간 (초 단위)'
    }
  }
};

const MotionSequenceSchema = {
  type: 'object',
  required: ['name', 'motions'],
  properties: {
    name: {
      type: 'string',
      description: '시퀀스 이름'
    },
    motions: {
      type: 'array',
      items: RobotMotionSchema,
      description: '모션 배열'
    }
  }
};

const SuccessResponseSchema = {
  type: 'object',
  properties: {
    status: { type: 'string' },
    message: { type: 'string' }
  }
};

const ErrorResponseSchema = {
  type: 'object',
  properties: {
    error: { type: 'string' }
  }
};

// --- 라우트 정의 시작 (나머지 라우트들은 원본 코드와 동일) ---

// 단일 모션 실행
app.post('/motion/execute', {
  schema: {
    tags: ['Motion'],
    description: '단일 모션 명령 실행',
    body: RobotMotionSchema,
    response: {
      200: SuccessResponseSchema,
      500: ErrorResponseSchema
    }
  }
}, async (request, reply) => {
  const motion = request.body;
  
  if (!await initSerial()) {
    return reply.code(500).send({ error: '시리얼 포트 연결 실패' });
  }
  
  try {
    const command = createCommand(motion);
    ser.write(command);
    console.log(command);
    
    if (motion.head_tilt || motion.head_pan) {
      console.log('head or tilt');
      await sleep(100);
      console.log(motion);
      const headCommand = createHeadCommand(motion);
      console.log(headCommand);
      ser.write(headCommand);
    }
    
    await sleep(motion.duration * 1000);
    return { status: 'success', message: '모션 실행 완료' };
  } catch (e) {
    return reply.code(500).send({ error: `모션 실행 실패: ${e.message}` });
  }
});

let pos_w = 90 // HEAD RIGHT-LEFT
let pos_h = 10 // HEAD UP-DOWN 

setInterval(()=>{
  pos_w = 90 // HEAD RIGHT-LEFT
  pos_h = 25
},60000)

let lastTime = 0

// head 처리
/*
app.get('/head/:name', {
  schema: {
    tags: ['head'],
    description: '헤드 위치 조정',
    params: {
      type: 'object',
      properties: {
        name: { type: 'string', description: 'up / down/ left/ right' }
      }
    },
    response: {
      200: MotionSequenceSchema,
      404: ErrorResponseSchema,
      500: ErrorResponseSchema
    }
  }
}, async (request, reply) => {
  try {
    const data = JSON.parse(await fs.readFile(MOTION_DATA_PATH, 'utf-8'));
    const { name } = request.params;
    
    let cmd = 0

    switch(name){
      case 'up':
        if (pos_h > 4)
          pos_h -= 3              
        break
      case 'down':
        if (pos_h < 33)
            pos_h += 3                
        break
      case 'left':
        if (pos_w > 11)
            pos_w -= 5  
        break;      
      case 'right':
        if (pos_w < 169)
            pos_w += 5                
        break
      }

      const headCommand = createHeadCommand({ head_tilt : pos_h, head_pan :pos_w})
      ser.write(headCommand);
    
  } catch (e) {
    return reply.code(500).send({ error: `조회 실패: ${e.message}` });
  }
});
*/
function getWifiIP() {
  const nets = os.networkInterfaces()
  const wifiRegex = /^(enp|wlo|wlx)/

  for (const name of Object.keys(nets)) {
    if (!wifiRegex.test(name)) continue
    for (const net of nets[name]) {
      if (net.family === 'IPv4' && !net.internal) {
        return net.address
      }
    }
  }
  return null
}

// ✅ API: Wi-Fi IP
app.get('/ip', async (request, reply) => {
  const ip = getWifiIP()

  return {
    ip,
    port: PORT,
    url: ip ? `http://${ip}:${PORT}` : null
  }
})

// 시퀀스 조회
app.get('/sequence/:name', {
  schema: {
    tags: ['Sequence'],
    description: '특정 시퀀스 조회',
    params: {
      type: 'object',
      properties: {
        name: { type: 'string', description: '시퀀스 이름' }
      }
    },
    response: {
      200: MotionSequenceSchema,
      404: ErrorResponseSchema,
      500: ErrorResponseSchema
    }
  }
}, async (request, reply) => {
  console.log("=================11")
  try {
    const data = JSON.parse(await fs.readFile(MOTION_DATA_PATH, 'utf-8'));
    const { name } = request.params;
    
  console.log("=================2",data[name])

    if (!data[name]) {
      return reply.code(404).send({ error: '시퀀스를 찾을 수 없습니다' });
    }
    

    console.log('here')
    return data[name];


  } catch (e) {
    return reply.code(500).send({ error: `조회 실패: ${e.message}` });
  }
});

app.get("/speak", async (request, reply) => {
  const text = request.query.text || "안녕하세요";

  const encodedText = encodeURIComponent(`"${text}"`);

  const options = {
    hostname: "127.0.0.1",
    port: 59530,
    path: `/v1/tts?text=${encodedText}`,
    method: "GET",
  };

  return new Promise((resolve, reject) => {
    const req = http.request(options, (res) => {
      const player = spawn("play", ["-"]); // stdin으로 재생

      res.pipe(player.stdin);

      player.on("close", (code) => {
        resolve({ status: "played" });
      });

      player.on("error", (err) => {
        reject(err);
      });
    });

    req.on("error", (err) => {
      reject(err);
    });

    req.end();
  });
});

let command = ""

app.get("/command", async (request, reply) => {

  const cmd = request.query.cmd || command;

  if(cmd != command)
    command = cmd

  return { result : true , data : cmd };
});

// 시퀀스 실행
app.get('/sequence/play/:name', {
  schema: {
    tags: ['Sequence'],
    description: '저장된 시퀀스 실행',
    params: {
      type: 'object',
      properties: {
        name: { type: 'string', description: '시퀀스 이름' }
      }
    },
    response: {
      200: SuccessResponseSchema,
      404: ErrorResponseSchema,
      500: ErrorResponseSchema
    }
  }
}, async (request, reply) => {
  if (!await initSerial()) {
    return reply.code(500).send({ error: '시리얼 포트 연결 실패' });
  }
  
  try {
    const data = JSON.parse(await fs.readFile(MOTION_DATA_PATH, 'utf-8'));
    const { name } = request.params;
    const sequence = data[name];
    
    if (!sequence) {
      return reply.code(404).send({ error: '시퀀스를 찾을 수 없습니다' });
    }
    
    for (const motion of sequence.motions) {
      const command = createCommand(motion);
      ser.write(command);
      await sleep(motion.duration * 1000);
    }
    
    return { status: 'success', message: `시퀀스 '${name}' 실행 완료` };
  } catch (e) {
    return reply.code(500).send({ error: `시퀀스 실행 실패: ${e.message}` });
  }
});

// 시퀀스 실행 (POST)
app.post('/sequence/execute', {
  schema: {
    tags: ['Sequence'],
    description: '모션 시퀀스 즉시 실행',
    body: MotionSequenceSchema,
    response: {
      200: SuccessResponseSchema,
      500: ErrorResponseSchema
    }
  }
}, async (request, reply) => {
  const sequence = request.body;
  
  if (!await initSerial()) {
    return reply.code(500).send({ error: '시리얼 포트 연결 실패' });
  }
  
  try {
    for (const motion of sequence.motions) {
      const command = createCommand(motion);
      ser.write(command);
      await sleep(motion.duration * 1000);
    }
    
    return { status: 'success', message: `시퀀스 '${sequence.name}' 실행 완료` };
  } catch (e) {
    return reply.code(500).send({ error: `시퀀스 실행 실패: ${e.message}` });
  }
});

// 시퀀스 저장
app.post('/sequence/save', {
  schema: {
    tags: ['Sequence'],
    description: '모션 시퀀스 저장',
    body: MotionSequenceSchema,
    response: {
      200: SuccessResponseSchema,
      500: ErrorResponseSchema
    }
  }
}, async (request, reply) => {
  const sequence = request.body;
  
  try {
    const data = JSON.parse(await fs.readFile(MOTION_DATA_PATH, 'utf-8'));
    data[sequence.name] = sequence;
    await fs.writeFile(MOTION_DATA_PATH, JSON.stringify(data, null, 2));
    return { status: 'success', message: `시퀀스 '${sequence.name}' 저장 완료` };
  } catch (e) {
    return reply.code(500).send({ error: `저장 실패: ${e.message}` });
  }
});

// 시퀀스 목록 조회
app.get('/sequence/list', {
  schema: {
    tags: ['Sequence'],
    description: '저장된 시퀀스 목록 조회',
    response: {
      200: {
        type: 'object',
        properties: {
          sequences: {
            type: 'array',
            items: { type: 'string' }
          }
        }
      },
      500: ErrorResponseSchema
    }
  }
}, async (request, reply) => {
  try {
    const data = JSON.parse(await fs.readFile(MOTION_DATA_PATH, 'utf-8'));
    return { sequences: Object.keys(data) };
  } catch (e) {
    return reply.code(500).send({ error: `목록 조회 실패: ${e.message}` });
  }
});

// 시퀀스 삭제
app.delete('/sequence/:name', {
  schema: {
    tags: ['Sequence'],
    description: '시퀀스 삭제',
    params: {
      type: 'object',
      properties: {
        name: { type: 'string', description: '시퀀스 이름' }
      }
    },
    response: {
      200: SuccessResponseSchema,
      404: ErrorResponseSchema,
      500: ErrorResponseSchema
    }
  }
}, async (request, reply) => {
  try {
    const data = JSON.parse(await fs.readFile(MOTION_DATA_PATH, 'utf-8'));
    const { name } = request.params;
    
    if (!data[name]) {
      return reply.code(404).send({ error: '시퀀스를 찾을 수 없습니다' });
    }
    
    delete data[name];
    await fs.writeFile(MOTION_DATA_PATH, JSON.stringify(data, null, 2));
    return { status: 'success', message: `시퀀스 '${name}' 삭제 완료` };
  } catch (e) {
    return reply.code(500).send({ error: `삭제 실패: ${e.message}` });
  }
});

// Heartbeat
app.get('/heartbeat', {
  schema: {
    tags: ['System'],
    description: '서버 하트비트 체크',
    response: {
      200: {
        type: 'object',
        properties: {
          result: { type: 'boolean' },
          data: { type: 'object' }
        }
      }
    }
  }
}, async (request, reply) => {
  console.log(state);
  return { result: true, data: state };
});

// MCR 초기화
app.get('/init_mcr', {
  schema: {
    tags: ['System'],
    description: 'MCR(Motor Control Robot) 시리얼 초기화',
    response: {
      200: {
        type: 'object',
        properties: {
          result: { type: 'boolean' }
        }
      },
      500: ErrorResponseSchema
    }
  }
}, async (request, reply) => {
  try {
    await initSerial();
    return { result: true };
  } catch (e) {
    return reply.code(500).send({ error: e.message });
  }
});

// 기본 모션
app.get('/motion', async (request, reply) => {
  const { name } = request.query;
  
  if (name) {
    return await app.inject({
      method: 'GET',
      url: `/sequence/play/${name}`
    });
  }
  
  // 기본 인사 모션
  const data_lower = Buffer.from([0xFF, 0x20, 0x00, 0x00, 0x00, 0x00]);
  const data_hello = Buffer.from([0xFE, 0x5A, 0x00, 0x50, 0x50, 0x00]);
  const data_hello_swing_left = Buffer.from([0xFE, 0x5A, 0x00, 0x50, 0x30, 0x00]);
  const data_hello_swing_right = Buffer.from([0xFE, 0x5A, 0x00, 0x50, 0x60, 0x00]);
  
  ser.write(data_lower);
  await sleep(1500);
  
  ser.write(data_hello);
  await sleep(1500);
  
  ser.write(data_hello_swing_left);
  await sleep(500);
  
  ser.write(data_hello_swing_right);
  await sleep(800);
  
  ser.write(data_hello_swing_left);
  await sleep(500);
  
  ser.write(data_hello_swing_right);
  await sleep(1800);
  
  ser.write(data_hello_swing_left);
  await sleep(500);
  
  ser.write(data_hello_swing_right);
  await sleep(800);
  
  ser.write(data_hello_swing_left);
  await sleep(500);
  
  ser.write(data_hello_swing_right);
  await sleep(800);
  
  ser.write(data_lower);
  await sleep(1500);
  
  return { result: true };
});


app.get('/stop', {
  schema: {
    tags: ['Gestures'],
    description: '⏹️ 정지',
    response: {
      200: {
        type: 'object',
        properties: {
          result: { type: 'boolean' }
        }
      }
    }
  }
}, async (request, reply) => {
  await move(Buffer.from([0xFF, 0x23, 0x00, 0x00, 0x43, 0x00]), 1000);
  await move(Buffer.from([0xFC, 0x15, 0x5A, 0x00, 0x00, 0x00]), 1000);
  return { result: true };
});

// WebSocket 제어
app.register(async function (fastify) {
  fastify.get('/ws/control', { websocket: true }, (connection, req) => {
    connection.socket.on('message', async (message) => {
      try {
        // message.toString() 대신 message.toString()을 사용합니다.
        const msg = JSON.parse(message.toString());
        
        if (msg.frame && Array.isArray(msg.frame)) {
          const frame = msg.frame;
          
          if (ser && ser.isOpen) {
            const ba = Buffer.from(frame);
            ser.write(ba);
          }
          
          connection.socket.send(JSON.stringify({ 
            status: 'ok', 
            sent: frame 
          }));
        }
      } catch (e) {
        console.error('WebSocket error:', e);
      }
    });
  });
});

// 모션 저장 / 플레이 API
app.post('/api/save_motion', async (request, reply) => {
  const frames = request.body;
  await fs.writeFile('motion_data.json', JSON.stringify(frames, null, 2));
  return { status: 'saved', count: frames.length };
});

app.post('/api/play_motion', async (request, reply) => {
  try {
    const frames = JSON.parse(await fs.readFile('motion_data.json', 'utf-8'));
    
    for (const frame of frames) {
      if (ser && ser.isOpen) {
        ser.write(Buffer.from(frame));
        await sleep(100);
      }
    }
    
    return { status: 'played', frames: frames.length };
  } catch (e) {
    return reply.code(404).send({ error: 'no saved motion' });
  }
});

// 서버 시작
const start = async () => {
  //let PORT = 8000;
  
  // 2. 최상위 await (top-level await)를 사용하여 포트 파일을 읽어옴 (ESM에서 지원됨)
  /*
  try {
    const portContent = await fs.readFile('port.txt', 'utf-8');
    PORT = parseInt(portContent.trim());
  } catch {
    console.log('port.txt not found, using default port 3000');
  }
  */

  try {
    await app.listen({ port: PORT, host: '0.0.0.0' });
    console.log('Loading Complete');
    console.log(`Server listening on http://0.0.0.0:${PORT}`);
    console.log(`Swagger UI: http://0.0.0.0:${PORT}/docs`);
  } catch (err) {
    app.log.error(err);
    process.exit(1);
  }
};

await start(); // 3. 최상위 await로 start 함수 호출
