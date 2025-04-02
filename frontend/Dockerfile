# # 1. 빌드 단계: Node.js Alpine 사용
# FROM node:22.13.1-alpine

# # 작업 디렉토리 설정
# WORKDIR /app

# # package.json과 package-lock.json을 복사
# COPY package*.json ./

# # 의존성 설치
# RUN npm install

# # ✅ react-scripts에 실행 권한 부여      #k8s에서 애러 뜸(권한 문제)
# RUN chmod +x node_modules/.bin/react-scripts


# # 소스 코드 복사
# COPY . .

# # 리액트 앱 빌드
# CMD ["npm", "start"]

# # global serve 설치
# RUN npm install -g serve

# # 정적 파일 서빙
# CMD ["serve", "-s", "build", "-l", "3000"]


FROM node:18-alpine

WORKDIR /app

# 이미 빌드된 파일만 복사
COPY build/ ./build/

# 서빙을 위한 serve 설치
RUN npm install -g serve

# 3000 포트로 서빙
EXPOSE 3003

# build 디렉토리 내용을 서빙
CMD ["serve", "-s", "build", "-l", "3000"]