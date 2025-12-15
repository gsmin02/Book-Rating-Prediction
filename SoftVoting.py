import os
import pandas

csv_path = "submission/"
submission_list = os.list(csv_path)

submission = pd.read_csv("test_ratings.csv")

for d in submission_list:
    submission['rating'] += pd.read_csv(csv_path + d)['rating']

submission /= len(submission_list)

submission.to_csv("submission.csv")