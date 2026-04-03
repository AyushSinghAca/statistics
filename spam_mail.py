import math
import pandas as pd
from collections import defaultdict

# Load Dataset
df = pd.read_csv("emails.csv")

# Convert Labels (as given in csv file)
df["s"] = df["spam"].map({1: "spam", 0: "not_spam"})

# Create Dataset exmpale: (email, spam or not_spam)
dataset = list(zip(df["text"], df["s"]))

# Break sentence into lowercase words
def preprocess(text):
    return str(text).lower().split()

# Train
def train(dataset):
    	#intialising variab
	spam_wc = defaultdict(int)         # save words along with how many times it is repeted
	not_spam_wc = defaultdict(int)
	spam_c = 0
	not_spam_c = 0
	vocab = set()

	for text, label in dataset:
		words = preprocess(text)
		vocab.update(words)


		if label == "spam":
			spam_c += 1
			for word in words:
				spam_wc[word] += 1
		else:
			not_spam_c += 1
			for word in words:
				not_spam_wc[word] += 1


	total = spam_c + not_spam_c

	# Prior generated from given datasets.
	p_spam= spam_c / total
	p_not_spam= not_spam_c / total

	return {
		"spam_word_count": spam_wc,
        	"not_spam_word_count": not_spam_wc,
        	"p_spam": p_spam,
        	"p_not_spam": p_not_spam,
        	"vocab": vocab,
		"spam_total_words": spam_c,
		"not_spam_total_words":not_spam_c,
	}
#  Checking Probability of New Given Text
def predict(model, text):
	words = preprocess(text)
	spam_score = math.log(model["p_spam"])  # Here it is including Prior  
	not_spam_score = math.log(model["p_not_spam"])
	V = len(model["vocab"])

	for word in words:
		# Laplace technique (Likehood for this Bayseian)
		spam_prob = (model["spam_word_count"][word] + 1) / (model["spam_total_words"] + V)
		not_spam_prob = (model["not_spam_word_count"][word] + 1) / (model["not_spam_total_words"] + V)

		spam_score += math.log(spam_prob)
		not_spam_score += math.log(not_spam_prob)

	return "SPAM" if spam_score > not_spam_score else "NOT SPAM"

# Train Model to get Prior
model = train(dataset)

# Test New mails
test_emails = [
    "Query regarding today class" ,"Win money now", "win money play this game",
 "Need Money", "Inquiry About Research Internship -2026"]

for email in test_emails:
    print(email, "->", predict(model, email))
