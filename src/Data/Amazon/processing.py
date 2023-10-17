import json
import random
from datetime import datetime
import os


file = open(os.getcwd()+"/src/All_Amazon_Review.json", "r")
out_file = open(os.getcwd()+"/src/Data/Amazon/amazon_review.csv", "a")



for line in file:
    try:
        obj  = json.loads(line)
        row_values = []

        overall = str(random.randint(1,5))
        if 'overall' in obj.keys():
            overall = str(obj['overall'])
        row_values.append(overall)

        vote = "0"
        if 'vote' in obj.keys():
            vote =str(obj['vote'])
        row_values.append(vote)
        
        time_ = ""
        if 'unixReviewTime' in obj.keys():
            time_=str(obj['unixReviewTime'])
        elif 'reviewTime' in obj.keys():
            date = datetime.strptime((obj['reviewTime']), '%m %d, %Y')
            time_ = str(date.timestamp())[:-2]
        row_values.append(time_)
        
        #synthetic dims
        row_values.append(str(random.randint(18,70))) # Age
        row_values.append(str(random.randint(0,28))) # Product category
        row_values.append(str(random.randint(1,195))) # Country
        
        out = ','.join(row_values)
        out= out+'\n'
        out_file.write(out)
        out_file.flush()

    except json.decoder.JSONDecodeError as e:
        continue


out_file.close()
file.close()