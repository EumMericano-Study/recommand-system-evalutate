import itertools

from surprise import accuracy
from collections import defaultdict


# surprise에서는 다양한 정확도 평가에 관한 알고리즘을 제공한다.   
# 정확도 평가보다 사용자 평가가 더 중요한 지표라고 했지만, 
# 수식으로서는 정확도 평가만큼 쉽게 제작할 수 있는 것도 없다.   

class RecommenderMetrics:

    def MAE(predictions):
        return accuracy.mae(predictions, verbose=False)

    def RMSE(predictions):
        return accuracy.rmse(predictions, verbose=False)

    # surprise는 주로 평가 예측을 하므로 
    # 먼저 추천 목록을 얻으려면 GetTopN 함수를 실행한다. 
    
    # 평가 예측 목록을 모두 받아서 사용자의 id값과 평가 항목들을 dictionary 형태로 연결해준다. 
    def GetTopN(predictions, n=10, minimumRating=4.0):
        # default Empty value : 키가 존재하지 않는 데이터에 접근할때 자동으로 Empty value를 설정해줌
        topN = defaultdict(list)


        for userID, movieID, actualRating, estimatedRating, _ in predictions:
            if (estimatedRating >= minimumRating):
                topN[int(userID)].append((int(movieID), estimatedRating))

        for userID, ratings in topN.items():
            ratings.sort(key=lambda x: x[1], reverse=True)
            topN[int(userID)] = ratings[:n]

        #dict 형태로 반환
        return topN

    # Hit Rate, 적중률을 계산하려면 사용자들의 id와 우선순위 목록이 키-값 형태로 저장된 dict와
    #           Test용 평점 데이터 세트를 매개변수로 전달해야 한다. 
    def HitRate(topNPredicted, leftOutPredictions):
        hits = 0
        total = 0

        # For each left-out rating
        for leftOut in leftOutPredictions:
            userID = leftOut[0]
            leftOutMovieID = leftOut[1]
            # Is it in the predicted top 10 for this user?
            hit = False
            for movieID, predictedRating in topNPredicted[int(userID)]:
                if (int(leftOutMovieID) == int(movieID)):
                    hit = True
                    break
            if (hit) :
                hits += 1

            total += 1

        # Compute overall precision
        return hits/total

    # 누적 적중률 (cHR)
    # 일반적인 HR에 적정값에 도달하지 못한 항목들을 제외하는 cut off value를 가진다.
    def CumulativeHitRate(topNPredicted, leftOutPredictions, ratingCutoff=0):
        hits = 0
        total = 0

        # For each left-out rating
        for userID, leftOutMovieID, actualRating, estimatedRating, _ in leftOutPredictions:
            # 실제 유저가 컷오프 점수보다 높은 평가를 한 데이터로만 hit을 계산
            if (actualRating >= ratingCutoff):
                # Is it in the predicted top 10 for this user?
                hit = False
                for movieID, predictedRating in topNPredicted[int(userID)]:
                    if (int(leftOutMovieID) == movieID):
                        hit = True
                        break
                if (hit) :
                    hits += 1

                total += 1

        # Compute overall precision
        return hits/total

    # 평가 적중률 (rHR)
    # 별도의 hit rate 계산과 동일하지만 각각의 평가값을 얻기위해
    # dict형태로 리턴하며, 각각의 지표들이 예측에 얼마나 영향을 미쳤는지 디테일하게 관찰할 수 있다.
    def RatingHitRate(topNPredicted, leftOutPredictions):
        hits = defaultdict(float)
        total = defaultdict(float)

        # For each left-out rating
        for userID, leftOutMovieID, actualRating, estimatedRating, _ in leftOutPredictions:
            # Is it in the predicted top N for this user?
            hit = False
            for movieID, predictedRating in topNPredicted[int(userID)]:
                if (int(leftOutMovieID) == movieID):
                    hit = True
                    break
            if (hit) :
                hits[actualRating] += 1

            total[actualRating] += 1

        # Compute overall precision
        for rating in sorted(hits.keys()):
            print (rating, hits[rating] / total[rating])


    # 평균 상호 적중률 (ARHR)
    # 순위의 역수를 더해 상위권 목록의 추천 퀄리티를 높인다.
    def AverageReciprocalHitRank(topNPredicted, leftOutPredictions):
        summation = 0
        total = 0
        # For each left-out rating
        for userID, leftOutMovieID, actualRating, estimatedRating, _ in leftOutPredictions:
            # Is it in the predicted top N for this user?
            hitRank = 0
            rank = 0
            for movieID, predictedRating in topNPredicted[int(userID)]:
                rank = rank + 1
                if (int(leftOutMovieID) == movieID):
                    hitRank = rank
                    break
            if (hitRank > 0) :
                summation += 1.0 / hitRank

            total += 1

        return summation / total

    # 유저 커버리지 계산
    # 임계값 이상의 "좋은" 추천의 수를 전체 유저수로 나눈 값
    def UserCoverage(topNPredicted, numUsers, ratingThreshold=0):
        hits = 0
        for userID in topNPredicted.keys():
            hit = False
            for movieID, predictedRating in topNPredicted[userID]:
                if (predictedRating >= ratingThreshold):
                    hit = True
                    break
            if (hit):
                hits += 1

        return hits / numUsers

    # 다양성 
    # 항목별 유사 메트릭을 매개변수로 받는다
    # 각 항목별 유사성의 평균값 S를 구한 뒤 1에서 빼주어 계산한다.
    # 실 데이터에서는 모든 조합을 평가하기 매우 복잡하기 때문에 표본 데이터를 이용하면 좋다.
    def Diversity(topNPredicted, simsAlgo):
        n = 0
        total = 0
        
        # 항목들을 모두 짝지어서 도출한 유사성 점수를 포함합니다
        simsMatrix = simsAlgo.compute_similarities()
        for userID in topNPredicted.keys():
            # 우선순위 항목에서 모든 항목을 짝지어준다.
            pairs = itertools.combinations(topNPredicted[userID], 2)
            # 이를 한쌍 씩 반복하여 각 쌍의 유사성을 찾는다.
            for pair in pairs:
                movie1 = pair[0][0]
                movie2 = pair[1][0]
                innerID1 = simsAlgo.trainset.to_inner_iid(str(movie1))
                innerID2 = simsAlgo.trainset.to_inner_iid(str(movie2))
                similarity = simsMatrix[innerID1][innerID2]
                total += similarity
                n += 1

        S = total / n
        return (1-S)

    def Novelty(topNPredicted, rankings):
        n = 0
        total = 0
        for userID in topNPredicted.keys():
            for rating in topNPredicted[userID]:
                movieID = rating[0]
                rank = rankings[movieID]
                total += rank
                n += 1
        return total / n
