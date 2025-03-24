# 1. 빌드 단계: Node.js Alpine 사용
FROM node:22.13.1-alpine

# 작업 디렉토리 설정
WORKDIR /app

# package.json과 package-lock.json을 복사
COPY package*.json ./

# 의존성 설치
RUN npm install

# 소스 코드 복사
COPY . .

# 리액트 앱 빌드
CMD ["npm", "start"]
