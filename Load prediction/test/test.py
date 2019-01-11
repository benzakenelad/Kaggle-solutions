import json
import requests
from fake_useragent import UserAgent

fake_ua = UserAgent()
self.user_agent = check_and_insert_user_agent(self, str(fake_ua.random))


s = requests.Session()
s.headers.update({
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Content-Length': '0',
    'Host': 'www.instagram.com',
    'Origin': 'https://www.instagram.com',
    'Referer': 'https://www.instagram.com/',
    'User-Agent': self.user_agent,
    'X-Instagram-AJAX': '1',
    'Content-Type': 'application/x-www-form-urlencoded',
    'X-Requested-With': 'XMLHttpRequest'
})
r = s.get('https://www.instagram.com/maayanbarnes/', headers="")
all_data = json.loads(r.text)



user_info = all_data['user']
follows = user_info['follows']['count']
follower = user_info['followed_by']['count']
follow_viewer = user_info['follows_viewer']
