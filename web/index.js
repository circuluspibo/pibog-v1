alert("Nice to meet you again! 2504162100")
// 버튼 클릭 효과 및 상태 변화 시뮬레이션
let isRecord = false
var gumStream;              //stream from getUserMedia()
var rec;                    //Recorder.js object
var input;                  //MediaStreamAudioSourceNode we'll be recording

// shim for AudioContext when it's not avb.
var AudioContext = window.AudioContext || window.webkitAudioContext;
var audioContext //new audio context to help us record
let lastTime = 0

function listen(){
  if (document.documentElement.requestFullscreen) 
    document.documentElement.requestFullscreen()


  if(!isRecord){
    
    console.log("Recording started")
    isRecord = true
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

function stt(blob){
  var fd = new FormData();
  fd.append("file", blob, 'voice.wav')

  fetch(`/v1/stt`,{ //?prompt=${query}&temp=${temp}&lang=en
    method: 'POST',
    body: fd
  }).then(async res=>{
    const result = await res.json()
    console.log('end---',result)
    $.query('[name=prompt]').value = result.data.text
    if($.query('[name=prompt]').value.length > 0)
      generate()    
  })

}

let keyStatus = {
    ArrowUp: false,
    ArrowDown: false,
    ArrowLeft: false,
    ArrowRight: false,
    PageUp: false,
    PageDown: false,
    Escape : false
};

const intervalTime = 100; // 키가 눌린 상태에서 반복하는 간격 (ms)

// 키가 눌렸을 때
document.addEventListener("keydown", (event) => {
    switch (event.key) {
        case "ArrowUp":
            if (!keyStatus.ArrowUp) {
                keyStatus.ArrowUp = true;
                startKeyRepeat("ArrowUp");
            }
            break;
        case "ArrowDown":
            if (!keyStatus.ArrowDown) {
                keyStatus.ArrowDown = true;
                startKeyRepeat("ArrowDown");
            }
            break;
        case "ArrowLeft":
            if (!keyStatus.ArrowLeft) {
                keyStatus.ArrowLeft = true;
                startKeyRepeat("ArrowLeft");
            }
            break;
        case "ArrowRight":
            if (!keyStatus.ArrowRight) {
                keyStatus.ArrowRight = true;
                startKeyRepeat("ArrowRight");
            }
            break;
        case "PageUp":
            if (!keyStatus.PageUp) {
                keyStatus.PageUp = true;
                startKeyRepeat("PageUp");
            }
            break;
        case "PageDown":
            if (!keyStatus.PageDown) {
                keyStatus.PageDown = true;
                startKeyRepeat("PageDown");
            }
            break;
    }
});

// 키가 떼어졌을 때
document.addEventListener("keyup", (event) => {

    cmd = ''

    switch (event.key) {
        case "ArrowUp":
            keyStatus.ArrowUp = false;
            break;
        case "ArrowDown":
            keyStatus.ArrowDown = false;
            break;
        case "ArrowLeft":
            keyStatus.ArrowLeft = false;
            break;
        case "ArrowRight":
            keyStatus.ArrowRight = false;
            break;
        case "PageUp":
            keyStatus.PageUp = false;
            break;
        case "PageDown":
            keyStatus.PageDown = false;
            break;
    }
});

// 키가 반복적으로 눌릴 때 실행될 함수
function startKeyRepeat(key) {
    const interval = setInterval(async () => {
        if (!keyStatus[key]) {
            clearInterval(interval); // 키가 떼어졌으면 반복 멈춤
        }
        switch (key) {
            case "ArrowUp":
                cmd = 'Move&x=1&y=0&z=0'
                break;
            case "ArrowDown":
                cmd = 'Move&x=-1&y=0&z=0'
                break;
            case "ArrowLeft":
                cmd = 'Move&x=0&y=1&z=0'
                break;
            case "ArrowRight":
                cmd = 'Move&x=0&y=-1&z=0'
                break;
            case "PageUp":
                cmd = 'Move&x=0&y=0&z=1'
                break;
            case "PageDown":
                cmd = 'Move&x=0&y=0&z=-1'
                break;
        }
        // 여기에 방향키 또는 페이지 업/다운에 대한 원하는 동작을 추가

        const response = await fetch(`/sport?cmd=${cmd}`)
        if (!response.ok) {
            throw new Error(`Response status: ${response.status}`)
        }
    
        const json = await response.json()
        console.log(cmd ,json)

    }, intervalTime);
    
}

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
                case 'move-up':
                    cmd = '/sport?cmd=Move&x=1&y=0&z=0'
                    break;
                case 'move-down':
                    cmd = '/sport?cmd=Move&x=-1&y=0&z=0'
                    break;                                        
                case 'move-left':
                    cmd = '/sport?cmd=Move&x=0&y=1&z=0'
                    break;
                case 'move-right':
                    cmd = '/sport?cmd=Move&x=0&y=-1&z=0'
                    break;
                case 'rotate-left':
                    cmd = '/sport?cmd=Move&x=0&y=0&z=1'
                    break;
                case 'rotate-right':
                    cmd = '/sport?cmd=Move&x=0&y=0&z=-1'
                    break;                    
                case 'tilt-up':
                    cmd = '/sport?cmd=BodyHeight&x=1&y=0&z=0' // get height
                    break;
                case 'tilt-down':
                    cmd = '/sport?cmd=BodyHeight&x=-1&y=0&z=0' // get height
                    break;        
                case 'tts-hello':
                    cmd = '/speech?text="안녕, 나는 파이독이야. 만나서 반가워요."&motion=Hello'
                    break;
                case 'tts-intro':
                    cmd = '/speech?text="안녕, 인텔과 함께하는 글로벌 챌린지에 온 것을 환영해."&motion=RiseSit'
                    break; 
                case 'tts-follow':
                    cmd = '/speech?text="안녕, 나는 인텔의 온 디바이스 A I를 활용하여 동작되는 로봇이야.."&motion=WiggleHips'
                    break;
                case 'tts-warn':
                    cmd = '/speech?text="나랑 부딪칠 수 있으니 조심히 피해줘."&motion=FrontPounce'
                    break;
                case 'tts-bye':
                    cmd = '/speech?text="다음에 다시 만나길 기대할께 안녕~!"&motion=Scrape'
                    break;
                case 'tts-poet':
                    cmd = '/speech?text="나는 강아지 로봇중에 최강인 파이독이라고 하지~ 사람을 물진 않아.?"&motion=FingerHeart'
                    break;    
                case 'mode':
                    if(mode == 'normal'){
                        mode = 'ai'
                        cmd = '/mode?value=ai'
                    } else {
                        mode = 'normal'
                        cmd = '/mode?value=normal'
                    }
                    break;     
                case 'connect':
                    fetch(`/connect`).then(async response=>{
                        if (!response.ok) {
                            throw new Error(`Response status: ${response.status}`)
                        }

                        const json = await response.json()
                        console.log('connect ok',json)

                        setTimeout(()=>{
                            document.getElementById('background-video').src = '/video_feed'
                        },5000)
                    })
                    break;
                default:
                    cmd = `/sport?cmd=${this.id}`
            }
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

        const peopleCount = document.getElementById('people-count');
        const objectCount = document.getElementById('object-count');
    

        peopleCount.textContent = data.cnt_live
        objectCount.textContent = data.cnt_object

        if(data.cnt_live > 0){
            const response = await fetch(`/color?value=red&warn=true`)
            if (!response.ok) {
                throw new Error(`Response status: ${response.status}`)
            }
        } else {
            const response = await fetch(`/color?value=cyan`)
            if (!response.ok) {
                throw new Error(`Response status: ${response.status}`)
            }           
        }


    }
    
    // 초기 시스템 상태 설정
    //updateSystemStatus();
    
    // 주기적으로 시스템 상태 업데이트 (5초마다)
    setInterval(updateSystemStatus, 30000);
    
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
    
});