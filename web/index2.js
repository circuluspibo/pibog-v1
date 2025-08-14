alert("Nice to meet you again! 2506260700")
// 버튼 클릭 효과 및 상태 변화 시뮬레이션
const list_tts = []
let audio = 0
let pose = ''
const poses = ['clamp','highFive','shakeHands_1','hug','lowWave','singleHandsUp','bothHandsUp']

let isRecord = false
var gumStream;              //stream from getUserMedia()
var rec;                    //Recorder.js object
var input;                  //MediaStreamAudioSourceNode we'll be recording

let multi = 1
let state = 'Walk_G1'

// shim for AudioContext when it's not avb.
var AudioContext = window.AudioContext || window.webkitAudioContext;
var audioContext //new audio context to help us record
let lastTime = 0

function listen(){
  //if (document.documentElement.requestFullscreen) 
  // document.documentElement.requestFullscreen()


  if(!isRecord){
    
    console.log("Recording started")
    isRecord = tze
    navigator.mediaDevices.getUserMedia({audio: true, video: false}).then(function(stream) {
      console.log("getUserMedia() success, stream created, initializing Recorder.js ...");

      audioContext = new AudioContext({sampleRate: 22050});

      // assign to gumStream for later use
      gumStream = stream;

      // use the stream
      input = audioContext.createMediaStreamSource(stream);

      // Create the Recorder object and configure to record mono sound (1 channel) Recording 2 channels will double the file size
      rec = new Recorder(input, {numChannels: 1})

      const options = {
        source: input,
        voice_stop: ()=> {
          console.log('voice_stop')
          isRecord = false
          console.log('record end')
          rec.stop(); //stop microphone access
          gumStream.getAudioTracks()[0].stop()
          rec.exportWAV(stt)
        }, 
        voice_start: function() {console.log('voice_start');}
       }; 
       
       // Create VAD
       //const vad = new VAD(options);      

      //start the recording process
      rec.record()

    }).catch(function(err) {
        //enable the record button if getUserMedia() fails
        console.log(err)
        isRecord = false
    });

  } else {

    isRecord = false
    console.log('record end')
    rec.stop(); //stop microphone access
    gumStream.getAudioTracks()[0].stop()
    rec.exportWAV(stt)
  }
}

function play(text){
    if(audio)
        audio.pause()

    audio = new Audio(`/v1/tts?voice=31&lang=ko&static=0&isPlay=0&text=${text}`);
    audio.play()
}

function playNext(chunk) {

    let cmd = ''

    if(chunk)
        list_tts.push(chunk)

    if(audio == 0 && list_tts.length > 0){
        if(pose == 'Release_Arm'){
            pose = poses[Math.floor(Math.random() * poses.length)]
            fetch(`/arm?cmd=${pose}`)
        } else {
            pose = 'Release_Arm'
            fetch(`/arm?cmd=${pose}`)
        }

        fetch(cmd)

        const text = list_tts.shift()
        audio = new Audio(`/v1/tts?voice=31&lang=ko&static=0&isPlay=0&text=${text}`);
        audio.play()

        audio.addEventListener('ended', () => {
            audio = 0
            playNext()  // Play next audio when current one ends
        })
    } 
}

  // Start playing the list
  playNext();

async function generate(prompt) {
  const response = await fetch(`http://127.0.0.1:59532/v1/txt2chat?prompt=${prompt}&isPlay=0`, {
    method: 'GET',
    headers: {
      'Accept': 'application/json'
    }
  });

  if (!response.ok) {
    console.error("HTTP error", response.status);
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let result = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value, { stream: true });
    result += chunk;

    // Update your UI with the streamed text here
    console.log(chunk); // for demo
    playNext(chunk)
    //document.getElementById("output").textContent += chunk;
  }

  console.log("Final result:", result);
}

function stt(blob){
  var fd = new FormData();
  fd.append("file", blob, 'voice.wav')

  fetch(`http://127.0.0.1:59532/v1/stt`,{ //?prompt=${query}&temp=${temp}&lang=en
    method: 'POST',
    body: fd
  }).then(async res=>{
    const result = await res.json()
    console.log('end---',result)
    const prompt = result.data
    if(prompt.length > 0)
      generate(prompt)    
  }).catch(e=>{
    console.error('err',e)
  })

}

function smooth(value){
    if(value < 0.5)
        return 0.5
    else if (value > 1)
        return 0.75
    else    
        return value
}

const keysPressed ={}

window.addEventListener('keydown', async (e) => {
    console.log(e.key)
    keysPressed[e.key] = true;

    /*
    if(e.key == 'Escape'){
        const response = await fetch(`/sport?cmd=Hello`)
        if (!response.ok) {
            throw new Error(`Response status: ${response.status}`)
        }

        const json = await response.json()
        console.log(cmd ,json)
    }
    */
});

window.addEventListener('keyup', (e) => {
    keysPressed[e.key] = false;
});

let intv = 0
const lastPressed = {}
let lastCmd = ''
let lastState = 'stop'

function getDirection() {

    const up = keysPressed['ArrowUp'];
    const down = keysPressed['ArrowDown'];
    const left = keysPressed['ArrowLeft'];
    const right = keysPressed['ArrowRight'];

    const rt_left = keysPressed['PageUp'];
    const rt_right = keysPressed['PageDown'];

    if( lastPressed['ArrowUp'] != keysPressed['ArrowUp'] ||lastPressed['ArrowDown'] != keysPressed['ArrowDown'] || 
        lastPressed['ArrowLeft'] != keysPressed['ArrowLeft'] ||lastPressed['ArrowRight'] != keysPressed['ArrowRight'] ||
        lastPressed['PageUp'] != keysPressed['PageUp'] ||lastPressed['PageDown'] != keysPressed['PageDown']){

        console.log('control',`/up ${up} /down ${down} /left ${left} /right ${right} / rt_left ${rt_left} /rt_right ${rt_right}`)
        lastPressed['ArrowUp'] = keysPressed['ArrowUp']
        lastPressed['ArrowDown'] = keysPressed['ArrowDown']
        lastPressed['ArrowLeft'] = keysPressed['ArrowLeft']
        lastPressed['ArrowRight'] = keysPressed['ArrowRight']
        lastPressed['PageUp'] = keysPressed['PageUp']
        lastPressed['PageDown'] = keysPressed['PageDown']
    }
        
    let lx = 0, ly = 0, rx = 0, ry=0
 

    // 방향 이동 처리
    if (up) { // ⬆ 북쪽
        ly = multi
        ry = multi
        lastState = 'up'
    } else if (down) { // ⬇ 남쪽
        lastState = 'down'        
        ly = -1 * multi
        ry = -1 * multi
    }
    
    if (left)  // ⬅ 서쪽
        lx = - 1 * smooth(multi)
    else if (right) // ➡ 동쪽
        lx = smooth(multi)

    // 회전 처리
    if (rt_left)
        rx = -1 * smooth(multi)
    else if (rt_right)
        rx = smooth(multi)
    
    // 명령어 생성 // pion 은 정지 명령도 필요하여 무조건 전송 - 다만 기존과 다를때만
    const cmd = `lx=${lx}&rx=${rx}&ly=${ly}&ry=${ry}`

    document.getElementById('log').value = cmd

    if(lastCmd != cmd){
        console.log('cmd',cmd)    
        lastCmd = cmd
        
        clearInterval(intv)
        
        if(cmd == 'lx=0&rx=0&ly=0&ry=0' && lastState == 'up' && multi > 1){
            lastState = 'stop'
            fetch(`/walkG1?lx=0&rx=0&ly=0.5&ry=0.5`)

            intv = setTimeout(function(){
                console.log('smooth stop....')
                fetch(`/walkG1?lx=0&rx=0&ly=0&ry=0`)
            },1000)
            
        } else
            fetch(`/walkG1?${cmd}`)
    }
}


function gameLoop() {
  getDirection()
  requestAnimationFrame(gameLoop)
}

gameLoop() // 루프 시작

// 글로벌 키 입력 감지
/*
document.addEventListener("keydown", (event) => {
    console.log(`Global Keydown: ${event.key}`);
});

document.addEventListener("keyup", (event) => {
    console.log(`Global Keyup: ${event.key}`);
});
*/

let mode = 'normal'

document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('object-speed').textContent = multi
    // 모든 기능 버튼에 클릭 이벤트 추가
    const allButtons = document.querySelectorAll('.function-button, .pad-button, .rotation-button, .mic-button');
    
    allButtons.forEach(button => {
        button.addEventListener('click', async function() {
            // 버튼을 누르는 시각적 효과
            this.style.transform = 'scale(0.95)';
            setTimeout(() => {
                if (this.classList.contains('function-button')) {
                    this.style.transform = 'translateY(-5px)';
                } else {
                    this.style.transform = 'scale(1)';
                }
            }, 100);
            
            // 명령어 로그
            console.log('명령어 실행:', this.id);

            cmd = ''

            if(this.id =='mic-button'){
                listen()
                return
            }

            switch(this.id){
                case 'rotation-center':
                case 'move-center':
                    cmd = '/walkG1?lx=0&rx=0&ly=0&ry=0'
                    break;                
                case 'move-up':
                    cmd = `/walkG1?lx=0&rx=0&ly=${1 * multi}&ry=${1 * multi}`
                    break;
                case 'move-down':
                    cmd = `/walkG1?lx=0&rx=0&ly=${-1 * multi}&ry=${-1 * multi}`
                    break;                                        
                case 'move-left':
                    cmd = `/walkG1?lx=${-1 * multi}&rx=0&ly=0&ry=0`
                    break;
                case 'move-right':
                    cmd = `/walkG1?lx=${1 * multi}&rx=0&ly=0&ry=0`
                    break;
                case 'rotate-left':
                    cmd = `/walkG1?lx=0&rx=${-1 * multi}&ly=0&ry=0`
                    //cmd = `/walkG1?lx=${-0.5 * multi}&rx=${-0.5 * multi}&ly=0&ry=0`
                    break;
                case 'rotate-right':
                    cmd = `/walkG1?lx=0&rx=${1 * multi}&ly=0&ry=0`
                    //cmd = `/walkG1?lx=${0.5 * multi}&rx=${0.5 * multi}&ly=0&ry=0`
                    break;                    
                case 'tilt-up':
                    cmd = `/walkG1?lx=0&rx=0&ly=${1.5 * multi}&ry=${1.5 * multi}`
                    break;
                case 'tilt-down':
                    cmd = `/walkG1?lx=0&rx=0&ly=${-1.5 * multi}&ry=${-1.5 * multi}`
                    break;        
                case 'tts-hello':
                    cmd = `/arm?cmd=shakeHands_1`
                    play("안녕? 나는 서큘러스의 파이온이라고 해. 만나서 반가워.")
                    break;
                case 'tts-intro':
                    cmd = `/arm?cmd=clamp`
                    play("서큘러스의 휴먼 인공지능기술과 만드로의 로봇 손 기술이 결합되었어.")
                    break; 
                case 'tts-follow':
                    cmd = `/arm?cmd=lowWave`
                    play("자 저를 따라 오세요!")
                    break;
                case 'tts-warn':
                    cmd = `/arm?cmd=highFive`
                    play("안녕하세요. 저와 함께 사진좀 찍어보실래요?")
                    break;
                case 'tts-bye':
                    cmd = `/arm?cmd=lowWave`
                    play("환영합니다. 저는 휴머노이드 로봇 파이온입니다.")
                    break;
                case 'tts-poet':
                    cmd = `/arm?cmd=Refuse`
                    play("저는 업무를 처리중이므로, 가까이 오시면 위험합니다.")
                    break;    
                case 'mode':
                    if(mode == 'Step_G1'){
                        mode = 'Stand_G1'
                        cmd = '/balanceG1?cmd=Stand_G1'
                        multi = 1
                    } else {
                        mode = 'Step_G1'
                        cmd = '/balanceG1?cmd=Step_G1'
                        multi = 2
                    }
                    break;     
                case 'connect':

                    fetch(`/start_collection`).then(async response=>{
                        if (!response.ok) {
                            throw new Error(`Response status: ${response.status}`)
                        }
   
                        document.getElementById('background-video').src = '/video_feed'
                        
                    })
              

                    fetch(`/connect2`).then(async response=>{
                        if (!response.ok) {
                            throw new Error(`Response status: ${response.status}`)
                        }

                        const json = await response.json()
                        console.log('connect ok',json)
                    })
                    break;
                case 'prepare':
                    cmd = '/prepare'
                    break;                    
                default:
                    if(this.classList.contains('hands'))
                        cmd = `/hand?cmd=${this.id}`
                    else if(this.classList.contains('states')){
                        state = this.id
                        multi = 0.5
                        cmd = `/stateG1?cmd=${this.id}`                    
                    } else
                        cmd = `/arm?cmd=${this.id}`
            }


            document.getElementById('log').value = cmd

            const response = await fetch(cmd)
            if (!response.ok) {
                throw new Error(`Response status: ${response.status}`)
            }
        
            const json = await response.json()
            console.log(cmd ,json)
            
            // 시스템 상태 실시간 변화 시뮬레이션
            //updateSystemStatus();
        });
    });
    
    // 마이크 버튼 클릭 시 녹음 시뮬레이션
    const micButton = document.getElementById('mic-button');
    let isRecording = false;
    
    micButton.addEventListener('click', function() {
        isRecording = !isRecording;
        
        if (isRecording) {
            this.innerHTML = '<i class="fas fa-stop"></i>';
            this.style.backgroundColor = 'var(--danger-color)';
            document.querySelector('.voice-input input').setAttribute('placeholder', '듣는 중...');
        } else {
            this.innerHTML = '<i class="fas fa-microphone"></i>';
            this.style.backgroundColor = 'var(--primary-color)';
            document.querySelector('.voice-input input').setAttribute('placeholder', '명령어를 입력하세요...');
            
            // 녹음 종료 후 입력창에 텍스트 표시 (시뮬레이션)
            setTimeout(() => {
                document.querySelector('.voice-input input').value = '앞으로 이동';
                
                // 잠시 후 명령어 실행 시뮬레이션
                setTimeout(() => {
                    document.querySelector('.voice-input input').value = '';
                    document.getElementById('move-up').click();
                }, 1000);
            }, 500);
        }
    });
    
    // 시스템 상태 변화 시뮬레이션
    async function updateSystemStatus() {
        const tempValue = document.getElementById('temp-value');
        const batteryValue = document.getElementById('battery-value');
        const cpuValue = document.getElementById('cpu-value');
        
        const tempFill = document.querySelector('.temperature-fill');
        const batteryFill = document.querySelector('.battery-fill');
        const cpuFill = document.querySelector('.cpu-fill');

        const response = await fetch(`/heartbeat`)
        if (!response.ok) {
            throw new Error(`Response status: ${response.status}`)
        }

        const data = (await response.json()).data

        console.log('system',data)
        
        // 온도: 40~50도 사이에서 랜덤하게 변화
        //const newTemp = Math.floor(Math.random() * 10) + 40;
        const newTemp = data.temp
        tempValue.textContent = `${newTemp}°C`;
        tempFill.style.width = `${newTemp}%`;
        
        // 배터리: 현재보다 0.5~1% 감소
        const currentBattery = parseInt(data.charge);
        const newBattery = Math.max(0, currentBattery - (Math.random() * 0.5 + 0.5));
        batteryValue.textContent = `${newBattery.toFixed(1)}%`;
        batteryFill.style.width = `${newBattery}%`;
        
        // CPU: 30~90% 사이에서 랜덤하게 변화
        //const newCpu = data.voltage//Math.floor(Math.random() * 60) + 30;
        cpuValue.textContent = `${data.voltage}%`;
        cpuFill.style.width = `${data.voltage}%`;
        
        // 배터리 색상 변화 (20% 이하일 때 빨간색으로 변경)
        if (newBattery <= 20) {
            batteryFill.style.backgroundColor = 'var(--danger-color)';
        } else {
            batteryFill.style.backgroundColor = 'var(--success-color)';
        }
        
        // 온도 색상 변화 (45도 이상일 때 더 진한 빨간색으로)
        if (newTemp >= 45) {
            tempFill.style.backgroundColor = '#ff0000';
        } else {
            tempFill.style.backgroundColor = 'var(--danger-color)';
        }
    }
    
    // 초기 시스템 상태 설정
    //updateSystemStatus();
    
    // 주기적으로 시스템 상태 업데이트 (5초마다)
    //setInterval(updateSystemStatus, 60000);
    
    // 사이버네틱 글리치 효과 랜덤 생성
    setInterval(() => {
        if (Math.random() > 0.8) {
            const glitchLine = document.querySelector('.glitch-line');
            glitchLine.style.top = `${Math.random() * 100}%`;
            glitchLine.style.opacity = '0.8';
            glitchLine.style.height = `${Math.random() * 10 + 2}px`;
            
            setTimeout(() => {
                glitchLine.style.opacity = '0';
            }, 100);
        }
    }, 2000);
    
    // 색상 테마 변경 기능
    const colorOptions = document.querySelectorAll('.color-option');
    const root = document.documentElement;
    
    const themeColors = {
        white: {
            primary: '#ffffff',
            secondary: '#cccccc',
            accent: '#e6e6e6',
            success: '#f0f0f0'
        },
        red: {
            primary: '#ff2f6b',
            secondary: '#cc0033',
            accent: '#ff668c',
            success: '#ff4d79'
        },
        yellow: {
            primary: '#ffcc00',
            secondary: '#ff9500',
            accent: '#ffe066',
            success: '#ffdb4d'
        },
        blue: {
            primary: '#00a2ff',
            secondary: '#0062ff',
            accent: '#66c7ff',
            success: '#33d0ff'
        },
        green: {
            primary: '#00ff7e',
            secondary: '#00b056',
            accent: '#5aff8d',
            success: '#33ff99'
        },
        cyan: {
            primary: '#00ffea',
            secondary: '#00b7c3',
            accent: '#00ffff',
            success: '#33fff0'
        },
        purple: {
            primary: '#9e00ff',
            secondary: '#7700cc',
            accent: '#bb4dff',
            success: '#aa33ff'
        }
    };
    
    colorOptions.forEach(option => {
        option.addEventListener('click', async function() {
            // 활성화된 색상 옵션 업데이트
            colorOptions.forEach(opt => opt.classList.remove('active'));
            this.classList.add('active');
            
            // 색상 테마 적용
            const color = this.getAttribute('data-color');
            const theme = themeColors[color];

            const response = await fetch(`/color?value=${color}`)
            if (!response.ok) {
                throw new Error(`Response status: ${response.status}`)
            }
        
            const json = await response.json()
            console.log(color ,json)

            
            root.style.setProperty('--primary-color', theme.primary);
            root.style.setProperty('--secondary-color', theme.secondary);
            root.style.setProperty('--accent-color', theme.accent);
            root.style.setProperty('--success-color', theme.success);
            root.style.setProperty('--current-theme', theme.primary);
            
            // 패널 그림자 색상 업데이트
            document.querySelectorAll('.voice-control, .status-panel, .color-panel, .counting-panel, .controls-panel').forEach(panel => {
                panel.style.boxShadow = `0 0 15px ${theme.primary}`;
                panel.style.borderColor = theme.primary;
            });
            
            // 제목 색상 업데이트
            document.querySelectorAll('.voice-control h3, .status-panel h3, .color-panel h3, .counting-panel h3, .controls-panel h3').forEach(title => {
                title.style.color = theme.primary;
            });
        });
    });
    
    // 밝기 조절 기능
    const brightnessSlider = document.getElementById('brightness-slider');
    const brightnessValue = document.getElementById('brightness-value');
    
    brightnessSlider.addEventListener('input', async function() {
        const value = this.value;
        brightnessValue.textContent = `${value}%`;

        const response = await fetch(`/brightness?value=${value}`)
        if (!response.ok) {
            throw new Error(`Response status: ${response.status}`)
        }
    
        const json = await response.json()
        console.log(value ,json)        
        
        // 화면 밝기 효과 적용 (CSS 필터 사용)
        document.getElementById('background-video').style.filter = `brightness(${value/100 * 0.8 + 0.2}) saturate(1.5)`;
    });

    // 볼륨 조절 기능
    const volumeSlider = document.getElementById('volume-slider');
    const volumeValue = document.getElementById('volume-value');
    const volumeIcon = document.querySelector('.fa-volume-up');
    
    volumeSlider.addEventListener('input', async function() {
        const value = this.value;
        volumeValue.textContent = `${value}%`;
        
        const response = await fetch(`/volume?value=${value}`)
        if (!response.ok) {
            throw new Error(`Response status: ${response.status}`)
        }
    
        const json = await response.json()
        console.log(value ,json)        

        // 볼륨 아이콘 변경
        if (value == 0) {
            volumeIcon.className = 'fas fa-volume-mute';
        } else if (value < 50) {
            volumeIcon.className = 'fas fa-volume-down';
        } else {
            volumeIcon.className = 'fas fa-volume-up';
        }
        
        // 실제 볼륨 조절은 구현되지 않음 (시뮬레이션)
    });
    
    // 카운팅 시뮬레이션 (랜덤하게 변화)
    const peopleCount = document.getElementById('people-count');
    const objectCount = document.getElementById('object-count');
    
    function updateCounts() {
        // 사람 수 변화 (2-8명 사이)
        const newPeopleCount = Math.floor(Math.random() * 7) + 2;
        peopleCount.textContent = newPeopleCount;
        
        // 사물 수 변화 (8-20개 사이)
        const newObjectCount = Math.floor(Math.random() * 13) + 8;
        objectCount.textContent = newObjectCount;
    }
    
    // 주기적으로 카운트 업데이트 (3초마다)
    setInterval(updateCounts, 3000);

   // 마우스 이벤트 처리
 
    const canvas = document.querySelector(".hud-container");

    let posX = window.innerWidth / 2;
    let posY = window.innerHeight / 2;

    // 포인터 잠금 요청
    canvas.addEventListener('click', () => {

        if (document.documentElement.requestFullscreen) 
            document.documentElement.requestFullscreen()

        canvas.requestPointerLock()
    })

    document.addEventListener('pointerlockchange', () => {
      if (document.pointerLockElement === canvas) {
        document.addEventListener("mousemove", updatePosition, false);
      } else {
        document.removeEventListener("mousemove", updatePosition, false);
      }
    });


    let rLevel = 0
    let rHeight = 0.5


    function smooth(value){
        if(value < 0.5)
            return 0.5
        else if (value > 1)
            return 1
        else    
            return value
    }

    async function setLevel(){
        if(rLevel == 0)
            rLevel == 1
        else 
            rLevel == 0

        const response = await fetch(`/sport?cmd=SpeedLevel&data=${rLevel}`)
        if (!response.ok) {
            throw new Error(`Response status: ${response.status}`)
        }

        const json = await response.json()
        console.log(cmd ,json)
    }

    async function setHeight(){
        if(rHeight == 0.5)
            rHeight == 0.3
        else
            rHeight == 0.5

        const response = await fetch(`/sport?cmd=BodyHeight&data=${rHeight}`)
        if (!response.ok) {
            throw new Error(`Response status: ${response.status}`)
        }

        const json = await response.json()
        console.log(cmd ,json)            
    }    

    function setSpeed(isUp){
        alert(state)
        if(state == 'Walk_G1' || state == 'Walk2_G1'){
            if(isUp){
                if(multi == 0.5)
                    multi = 1
                else if(multi == 1)
                    multi = 1.5
                else if(multi == 1.5)
                    multi = 2
                else
                    multi = 2
            } else {
                if(multi == 2)
                    multi = 1.5
                else if(multi == 1.5)
                    multi = 1
                else if(multi == 1)
                    multi = 0.5
                else
                    multi = 0.5
            }
        } else {
            if(isUp){
                if(multi < 1)
                    multi = 1
                else if(multi < 1.5)
                    multi = 1.5
                else if(multi < 2)
                    multi = 2.5
                else
                    multi = 3
            } else {
                if(multi < 1.5)
                    multi = 1
                else if(multi < 2)
                    multi = 1.5
                else if(multi < 3)
                    multi = 2
                else
                    multi = 1
            }            
        }

        document.getElementById('object-speed').textContent = multi
    }

    // 속도 조정 (geer)
    async function updatePosition(e) {
        const dx = e.movementX;
        const dy = e.movementY;

        // 좌표 업데이트
        posX += dx;
        posY += dy;

        // 화면 경계 제한
        posX = Math.max(0, Math.min(window.innerWidth, posX));
        posY = Math.max(0, Math.min(window.innerHeight, posY));

        if(dy > 0) // down
            multi = 0.5
        else if(dy < 0) // up
            multi = 2

        if(dx > 0) // right
            multi = 1.5 //setHeight(true)
        else if(dx < 0) // left
            multi = 1    

        document.getElementById('object-speed').textContent = multi
        // 방향 감지

        /*
        keysPressed['ArrowRight'] = false
        keysPressed['ArrowLeft'] = false
        keysPressed['ArrowDown'] = false
        keysPressed['ArrowUp'] = false        
        let key = '-';
        key = dx > 0 ? 'ArrowRight' : 'ArrowLeft';

        if(dx > 0)
            keysPressed['ArrowRight'] = true
        else if(dx < 0)
            keysPressed['ArrowLeft'] = true
        
        key = dy > 0 ? 'ArrowDown' : 'ArrowUp';

        if(dy > 0)
            keysPressed['ArrowDown'] = true
        else if(dy < 0)
            keysPressed['ArrowUp'] = true
        */        
    }


});