# LawBot
대화형 유사 판례 검색기능

# 판례 데이터
[대법원 API](https://open.law.go.kr/LSO/openApi/guideList.do)를 이용하여 대법원 판례 약 85,000개를 수집했습니다. 추후 전처리를 통해 약 60,000개의 판례를 모델에 사용했습니다.


# 파이프라인
![](image/pipeline.png)

- 사용자의 현 상황을 sentence bert로 임베딩한 후, 임베딩된 판례 데이터들과 코사인 유사도를 이용해 가장 유사한 판례를 찾습니다.
- 해당 판례를 LLM을 통해 요약하고, 사용자의 입력 상황에 맞게 판단을 내려립니다.


# 결과
웹 데모를 사용해보실려면 아래 코드를 입력하세요.

```shell
cd polyglot-jax-inference
python app.py
```
![](image/demo.png)

# 평가
![](image/evaluation.png)
<!-- Keyword Search와 Retrieve only의 경우 판례문 그 자체를 출력으로 하여, G-EVAL을 제외하고는 측정하지 않았습니다.

LegalQA는 기존 한국어용 법률 질의응답 모델로, 판례 대신 사전 질의응답에서 답변을 생성하는 방식입니다.

LLM을 사용한 경우 Readability가 증가하였으며, 판례 데이터를 넣어주었을 경우 Retrieval 성능 향상이 있었습니다.

다만 챗봇의 성능을 정량적으로 평가하기에 한계가 있으며, 이는 G-EVAL도 마찬가지 입니다.

전체적인 경향성으로 보아, 본 프로젝트의 접근법이 판례를 검색하여 보여주기에 가장 적절할 것으로 예상합니다. -->

