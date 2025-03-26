// utils/microphoneHandler.js
let mediaRecorder;

export const handleMicrophoneClick = (setCurrentMessage, setIsRecording) => {
    console.log("마이크 버튼이 클릭되었습니다.");

    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
        console.log("녹음 중단");
        setIsRecording(false);
        return;
    }

    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(stream => {
                mediaRecorder = new MediaRecorder(stream);
                mediaRecorder.start();
                setIsRecording(true);

                console.log("녹음 시작");

                const audioChunks = [];
                mediaRecorder.ondataavailable = event => {
                    audioChunks.push(event.data);
                };

                mediaRecorder.onstop = () => {
                    setIsRecording(false);
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });

                    // 오디오 데이터를 서버로 전송
                    const formData = new FormData();
                    formData.append('audio', audioBlob, 'recording.wav');

                    fetch('/api/upload-audio/', {
                        method: 'POST',
                        body: formData
                    })
                    .then(response => response.json())
                    .then(data => {
                        console.log("오디오 업로드 성공 및 텍스트 변환 결과:", data.text);
                        
                        // STT 결과를 입력창에 반영
                        if (setCurrentMessage && data.text) {
                            setCurrentMessage(data.text);
                        }
                    })
                    .catch(error => {
                        console.error("오디오 업로드 중 에러:", error);
                        setIsRecording(false); // 에러 발생 시에도 녹음 상태 해제
                    });

                    console.log("녹음 종료 및 서버로 전송");
                };
            })
            .catch(error => {
                console.error("녹음 중 에러:", error);
                setIsRecording(false);
            });
    } else {
        console.error("브라우저가 오디오 녹음을 지원하지 않습니다.");
        setIsRecording(false);
    }
};