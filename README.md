# HEADPOS Modification 

## Functions
고개 들면 점수 회복, 고개 내리면 점수 하락 

## Variables for Adjustment
1. self.PITCH_THRESHOLD = -10.0
2. self.DECAY_RATE = 0.1 # 점수 감소 rate
3. self.RECOVERY_RATE = 0.05 # 점수 회복 rate
4. self.HEAD_DOWN_THRESHOLD = 2.0 # 고개를 2초 이상 숙이고 있을 시 점수 하락 시작 
5. self.HEAD_UP_THRESHOLD = 2.0 # 고개를 2초 이상 들고 있을 시 점수 회복 시작 

## Todo 
1. 점수 회복 및 하락 rate 조정 