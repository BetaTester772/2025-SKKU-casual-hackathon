from kokoro import KPipeline
import soundfile as sf
import numpy as np

pipeline = KPipeline(lang_code='a')

text = '''Hello! Today I'll share some heartwarming family news.

First, Dad is grilling delicious meat and enjoying happy moments with the family. It looks even tastier served with kimchi.

My brother is having fun with friends after finishing a tough exam. He's taking a break to recharge his emotions in his purple room and has even made plans for an outing with Dad next time.

Mom is spending another warm day today, thinking of her family with a fragrant heart. It's truly beautiful to see how the family supports each other.

And the youngest enjoyed a stroll while admiring the beautiful city nightscape. The lights sparkling on the water were magical, they said. They're also enthusiastically participating in a hackathon, taking on a new challenge.

This is our family, each finding happiness and fulfillment in their own way. Have a healthy and joyful day today!

Translated with DeepL.com (free version)'''

generator = pipeline(text, voice='af_heart')

# 오디오 조각들을 하나의 배열에 이어붙이기
all_audio = []
for i, (gs, ps, audio) in enumerate(generator):
    print(i, gs, ps)
    all_audio.append(audio)

# numpy 배열로 합치기
merged_audio = np.concatenate(all_audio)

# 하나의 파일로 저장
sf.write("family_news.wav", merged_audio, 24000)
