# pkg_list.py
import os
import sys
import pkgutil

# 1) 실제 작업 디렉토리
print("1) 현재 작업 디렉토리:", os.getcwd())

# 2) 파이썬이 모듈 검색 시 뒤지는 경로 목록 (sys.path)
print("2) sys.path:")
for p in sys.path:
    print("   ", p)

# 3) 디렉토리 내 파일/폴더 현황
print("3) 디렉토리 항목:")
for name in os.listdir('.'):
    print("   ", name)

# 4) import 가능하다고 인식되는 패키지/모듈
print("\n4) import 가능 패키지·모듈:")
for finder, name, ispkg in pkgutil.iter_modules(path=['.']):
    kind = "패키지" if ispkg else "모듈"
    print(f"   - {name} ({kind})")