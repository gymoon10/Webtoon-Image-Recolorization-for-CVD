from bs4 import BeautifulSoup
import requests
import os
cwd = os.getcwd()

                    #"https://comic.naver.com/webtoon/list?titleId=748105", #독립일기
                    #"https://comic.naver.com/webtoon/list?titleId=641253", #외모지상주의
                    #"https://comic.naver.com/webtoon/list?titleId=183559", #신의 탑
                    #"https://comic.naver.com/webtoon/list?titleId=703846", #여신강림
                    #"https://comic.naver.com/webtoon/list?titleId=570503", #연애혁명
                    #"https://comic.naver.com/webtoon/list?titleId=597447", #프리드로우
                    #"https://comic.naver.com/webtoon/list?titleId=648419", #뷰티풀군바리
                    #"https://comic.naver.com/webtoon/list?titleId=389848", #헬퍼
                    #"https://comic.naver.com/webtoon/list?titleId=602910", #윈드브레이커
webtoon_url_list =["https://comic.naver.com/webtoon/list?titleId=318995", #갓오브하이스쿨
                   "https://comic.naver.com/webtoon/list?titleId=557672", #기기괴괴
                   "https://comic.naver.com/webtoon/list?titleId=695796", #내일
                   "https://comic.naver.com/webtoon/list?titleId=654774", #소녀의세계
                   "https://comic.naver.com/webtoon/list?titleId=683496", #신도림
                   "https://comic.naver.com/webtoon/list?titleId=667573", #연놈
                   "https://comic.naver.com/webtoon/list?titleId=711422", #삼국지톡
                   "https://comic.naver.com/webtoon/list?titleId=616239", #윌유메리미
                   "https://comic.naver.com/webtoon/list?titleId=702608", #랜덤채팅의 그녀
                   "https://comic.naver.com/webtoon/list?titleId=552960", #더게이머
                   "https://comic.naver.com/webtoon/list?titleId=655746" #마법스크롤상인지오
                  ]
#page_num_list = [18,40,55,22,41,45,34,19,40,56,41,26,33,28,34,40,84,24,43,33] # 전체 화 다 가져오기

for v,w in enumerate(webtoon_url_list):    
    # for i in range(page_num_list[v]) # 전체 화 다 가져오기
    num = 0
    
    for i in range(18): # 최대 180화까지만
        url = w +"&page={0}".format(i) 
        # 크롤링 우회
        headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.61 Safari/537.36'}
        html = requests.get(url, headers = headers)
        result = BeautifulSoup(html.content, "html.parser")

        webtoonName = result.find("span", {"class", "wrt_nm"}).parent.get_text().strip().split('\n')

        if os.path.isdir(os.path.join(cwd,  webtoonName[0])) == False: 
            os.mkdir(webtoonName[0])

        print(webtoonName[0] + " folder created successfully!")
        title = result.findAll("td", {"class", "title"})        
        #image_name

        for t in title:
            #각 회차별 url
            url ="https://comic.naver.com" + t.a['href']

            #헤더 우회해서 링크 가져오기
            html2 = requests.get(url, headers = headers) 
            result2 = BeautifulSoup(html2.content, "html.parser") 

            # webtoon image 찾기
            webtoonImg = result2.find("div", {"class", "wt_viewer"}).findAll("img")
            for i in webtoonImg:
                saveName = os.path.join(cwd,  webtoonName[0]) + "/" + str(num) + ".jpg"
                with open(saveName, "wb") as file:
                    src = requests.get(i['src'], headers = headers) 
                    file.write(src.content) #
                num += 1
            print((t.text).strip() + " saved completely!") 