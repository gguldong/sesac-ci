document.addEventListener('DOMContentLoaded', function() {
    const divider = document.querySelector('.divider');
    const dividerAfter = document.querySelector('.divider::after'); // 가상 요소 선택

    // 초기 상태 설정
    divider.style.height = '0'; // 선의 높이를 0으로 설정
    divider.style.opacity = '0'; // 선을 투명하게 설정
    divider.style.transition = 'none'; // 초기 상태에는 transition 효과 제거

    // Intersection Observer 설정
    const observer = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                // 선이 나타날 때 애니메이션 적용
                divider.style.transition = 'height 1.2s ease, opacity 1.2s ease'; // transition 효과 추가
                divider.style.height = '160px'; // 선의 높이를 원래대로 복원
                divider.style.opacity = '1'; // 선을 보이게 함

                // 가상 요소 애니메이션 (화살표 효과)
                dividerAfter.style.transition = 'transform 0.6s ease-in-out 1.2s, opacity 0.6s ease-in-out 1.2s'; // transition 추가 (지연 시간 설정)
                dividerAfter.style.transform = 'translate(-50%) translateY(10px)'; // 아래로 이동
                dividerAfter.style.opacity = '1'; // 보이게 함

                observer.unobserve(divider); // 한 번만 실행되도록 관찰 중지
            }
        });
    });

    observer.observe(divider); // 선 관찰 시작
});

document.addEventListener('DOMContentLoaded', function() {
    const infoBoxes = document.querySelector('.info-boxes');
    const infoBoxList = infoBoxes.querySelectorAll('.info-box');

    const observer = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                // info-boxes 요소가 화면에 나타났을 때 애니메이션 실행
                infoBoxList.forEach((box, index) => {
                    setTimeout(() => {
                        box.style.opacity = '1';
                        box.style.transform = 'translateY(0)';
                    }, 500 * index); // 순차적인 지연 시간 조절
                });
            } else {
                // info-boxes 요소가 화면에서 사라졌을 때 초기 상태로 되돌림
                infoBoxList.forEach(box => {
                    box.style.opacity = '0';
                    box.style.transform = 'translateY(50px)';
                });
            }
        });
    });

    observer.observe(infoBoxes);
});