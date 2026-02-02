# security.py

from passlib.context import CryptContext

# Bcrypt 암호화 방식 사용
pwd_context = CryptContext(
    schemes=["bcrypt"], 
    deprecated="auto"
)
# 🚨 주의: 이전에 추가했던 set_backend 코드와 관련 import를 모두 제거했습니다.

def verify_password(plain_password, hashed_password):
    """
    일반 비밀번호와 해시된 비밀번호를 비교합니다.
    (주의: bcrypt 72자 제한으로 인해, 원본 비밀번호도 72자로 잘라서 비교)
    """
    return pwd_context.verify(plain_password[:72], hashed_password)

def get_password_hash(password):
    """
    일반 비밀번호를 해시값으로 변환합니다.
    (중요) bcrypt는 72바이트(글자) 제한이 있으므로, 그 이상은 잘라냅니다.
    """
    return pwd_context.hash(password[:72])