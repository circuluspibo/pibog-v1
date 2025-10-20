const fastify = require('fastify')({ logger: true });
const fastifyStatic = require('@fastify/static');
const fastifyCors = require('@fastify/cors');
const fastifyWebsocket = require('@fastify/websocket');
const fastifySwagger = require('@fastify/swagger');
const fastifySwaggerUi = require('@fastify/swagger-ui');
const { SerialPort } = require('serialport');
const fs = require('fs').promises;
const path = require('path');
const crypto = require('crypto');

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
fastify.register(fastifyCors, {
  origin: '*',
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: '*'
});

// Swagger 설정
fastify.register(fastifySwagger, {
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

fastify.register(fastifySwaggerUi, {
  routePrefix: '/documentation',
  uiConfig: {
    docExpansion: 'list',
    deepLinking: false
  },
  staticCSP: true,
  transformStaticCSP: (header) => header
});

// 정적 파일 서빙
fastify.register(fastifyStatic, {
  root: path.join(__dirname, 'web'),
  prefix: '/web/'
});

fastify.register(fastifyStatic, {
  root: path.join(__dirname, 'webfonts'),
  prefix: '/webfonts/',
  decorateReply: false
});

// WebSocket 지원
fastify.register(fastifyWebsocket);

// 서버 시작시 초기화
fastify.addHook('onReady', async () => {
  await initSerial();
  
  // 모션 데이터 파일이 없으면 생성
  try {
    await fs.access(MOTION_DATA_PATH);
  } catch {
    await fs.writeFile(MOTION_DATA_PATH, JSON.stringify({}));
  }
});

// 서버 종료시 시리얼 포트 닫기
fastify.addHook('onClose', async () => {
  if (ser && ser.isOpen) {
    ser.close();
  }
});

// 라우트 정의
fastify.get('/', {
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

// 포트 번호 읽기 (포트 파일이 있다면)
let PORT = 3000;
try {
  const portContent = await fs.readFile('port.txt', 'utf-8');
  PORT = parseInt(portContent.trim());
} catch {
  console.log('port.txt not found, using default port 3000');
}

// Swagger 스키마 정의
const RobotMotionSchema = {
  type: 'object',
  required: ['arm', 'head_tilt', 'head_pan', 'shoulder_front', 'shoulder_side', 'elbow_front', 'elbow_side', 'finger', 'duration'],
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
      maximum: 6,
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

// 단일 모션 실행
fastify.post('/motion/execute', {
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

// 시퀀스 조회
fastify.get('/sequence/:name', {
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
  try {
    const data = JSON.parse(await fs.readFile(MOTION_DATA_PATH, 'utf-8'));
    const { name } = request.params;
    
    if (!data[name]) {
      return reply.code(404).send({ error: '시퀀스를 찾을 수 없습니다' });
    }
    
    return data[name];
  } catch (e) {
    return reply.code(500).send({ error: `조회 실패: ${e.message}` });
  }
});

// 시퀀스 실행
fastify.get('/sequence/play/:name', {
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
fastify.post('/sequence/execute', {
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
fastify.post('/sequence/save', {
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
fastify.get('/sequence/list', {
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
fastify.delete('/sequence/:name', {
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
fastify.get('/heartbeat', {
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
fastify.get('/init_mcr', {
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
fastify.get('/motion', async (request, reply) => {
  const { name } = request.query;
  
  if (name) {
    return await fastify.inject({
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

// 제스처 API들
fastify.get('/kiss_one_hand', {
  schema: {
    tags: ['Gestures'],
    description: '💋 한 손 키스 날리기',
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
  await move(Buffer.from([0xFE, 0x60, 0x30, 0x50, 0x50, 0x01]), 800);
  await move(Buffer.from([0xFE, 0x60, 0x30, 0x50, 0x50, 0x03]), 800);
  await move(Buffer.from([0xFE, 0x60, 0x30, 0x50, 0x50, 0x06]), 1000);
  return { result: true };
});

fastify.get('/carry_with_one_hand', {
  schema: {
    tags: ['Gestures'],
    description: '📦 한 손으로 짐 나르기',
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
  await move(Buffer.from([0xFE, 0x70, 0x10, 0x40, 0x50, 0x03]), 1500);
  return { result: true };
});

fastify.get('/carry_box_both_hands', {
  schema: {
    tags: ['Gestures'],
    description: '📥 양손 박스 운반',
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
  await move(Buffer.from([0xFF, 0x70, 0x10, 0x50, 0x50, 0x03]), 1500);
  return { result: true };
});

fastify.get('/make_heart', {
  schema: {
    tags: ['Gestures'],
    description: '💖 하트 만들기',
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
  await move(Buffer.from([0xFF, 0x90, 0x30, 0x60, 0x40, 0x06]), 2000);
  return { result: true };
});

fastify.get('/wave_in_front_of_face', {
  schema: {
    tags: ['Gestures'],
    description: '👋 얼굴 앞 손 흔들기',
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
  const base = Buffer.from([0xFE, 0x60, 0x10, 0x40, 0x40, 0x00]);
  const left = Buffer.from([0xFE, 0x60, 0x10, 0x40, 0x30, 0x00]);
  const right = Buffer.from([0xFE, 0x60, 0x10, 0x40, 0x50, 0x00]);
  
  await move(base, 1000);
  for (let i = 0; i < 10; i++) {
    await move(left, 400);
    await move(right, 400);
  }
  return { result: true };
});

fastify.get('/wave_above_head', {
  schema: {
    tags: ['Gestures'],
    description: '🙆 머리 위 손 흔들기',
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
  const base = Buffer.from([0xFE, 0xA0, 0x30, 0x50, 0x40, 0x00]);
  const left = Buffer.from([0xFE, 0xA0, 0x30, 0x50, 0x30, 0x00]);
  const right = Buffer.from([0xFE, 0xA0, 0x30, 0x50, 0x50, 0x00]);
  
  await move(base, 1000);
  for (let i = 0; i < 5; i++) {
    await move(left, 1000);
    await move(right, 1000);
  }
  return { result: true };
});

fastify.get('/raise_left_hand', {
  schema: {
    tags: ['Gestures'],
    description: '🙋‍♀️ 왼손 들기',
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
  await move(Buffer.from([0xFD, 0x90, 0x00, 0x50, 0x30, 0x00]), 1500);
  return { result: true };
});

fastify.get('/raise_right_hand', {
  schema: {
    tags: ['Gestures'],
    description: '🙋‍♂️ 오른손 들기',
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
  await move(Buffer.from([0xFE, 0x90, 0x00, 0x50, 0x30, 0x00]), 1500);
  return { result: true };
});

fastify.get('/make_x_pose', {
  schema: {
    tags: ['Gestures'],
    description: '❌ 팔 X자 모양 만들기',
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
  await move(Buffer.from([0xFF, 0x80, 0x40, 0x50, 0x20, 0x00]), 2000);
  return { result: true };
});

fastify.get('/clap', {
  schema: {
    tags: ['Gestures'],
    description: '👏 박수치기',
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
  const close = Buffer.from([0xFF, 0x60, 0x10, 0x60, 0x30, 0x03]);
  const open = Buffer.from([0xFF, 0x60, 0x10, 0x60, 0x60, 0x06]);
  
  for (let i = 0; i < 5; i++) {
    await move(close, 1000);
    await move(open, 1000);
  }
  return { result: true };
});

fastify.get('/spread_arms', {
  schema: {
    tags: ['Gestures'],
    description: '👐 팔 벌리기',
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
  await move(Buffer.from([0xFF, 0x40, 0x80, 0x30, 0x30, 0x00]), 1500);
  return { result: true };
});

fastify.get('/raise_both_arms', {
  schema: {
    tags: ['Gestures'],
    description: '🙌 양손 위로 들기',
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
  await move(Buffer.from([0xFF, 0xA0, 0x10, 0x50, 0x40, 0x00]), 2000);
  return { result: true };
});

fastify.get('/default_pose', {
  schema: {
    tags: ['Gestures'],
    description: '🧍 기본자세 (팔 내림)',
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
  await move(Buffer.from([0xFF, 0x20, 0x00, 0x00, 0x00, 0x00]), 1500);
  return { result: true };
});

fastify.get('/left_arm_out_right_up', {
  schema: {
    tags: ['Gestures'],
    description: '🔀 왼팔 펴고 오른팔 위로',
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
  const left = Buffer.from([0xFD, 0x50, 0x70, 0x30, 0x40, 0x00]);
  const right = Buffer.from([0xFE, 0xA0, 0x10, 0x50, 0x30, 0x00]);
  await move(left, 500);
  await move(right, 1500);
  return { result: true };
});

fastify.get('/right_arm_out_left_up', {
  schema: {
    tags: ['Gestures'],
    description: '🔁 오른팔 펴고 왼팔 위로',
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
  const right = Buffer.from([0xFE, 0x50, 0x70, 0x30, 0x40, 0x00]);
  const left = Buffer.from([0xFD, 0xA0, 0x10, 0x50, 0x30, 0x00]);
  await move(right, 500);
  await move(left, 1500);
  return { result: true };
});

fastify.get('/stop', {
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
  await move(Buffer.from([0xFF, 0x00, 0x00, 0x00, 0x00, 0x00]), 1000);
  return { result: true };
});

// WebSocket 제어
fastify.register(async function (fastify) {
  fastify.get('/ws/control', { websocket: true }, (connection, req) => {
    connection.socket.on('message', async (message) => {
      try {
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
fastify.post('/api/save_motion', async (request, reply) => {
  const frames = request.body;
  await fs.writeFile('motion_data.json', JSON.stringify(frames, null, 2));
  return { status: 'saved', count: frames.length };
});

fastify.post('/api/play_motion', async (request, reply) => {
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
  try {
    await fastify.listen({ port: PORT, host: '0.0.0.0' });
    console.log('Loading Complete');
    console.log(`Server listening on http://0.0.0.0:${PORT}`);
  } catch (err) {
    fastify.log.error(err);
    process.exit(1);
  }
};

start();
