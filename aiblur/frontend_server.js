const express = require('express');
const path = require('path');
const app = express();

// 정적 파일 제공
app.use(express.static(path.join(__dirname, 'frontend')));

// SPA 라우팅: 모든 요청을 index.html로
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'frontend', 'index.html'));
});

const PORT = 8003;
app.listen(PORT, '0.0.0.0', () => {
  console.log(`AIBlur Frontend running on http://0.0.0.0:${PORT}`);
});
