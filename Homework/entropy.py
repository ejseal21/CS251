import scipy.stats as stats

def entropy(class_counts):
	probs = []
	for i in range(max(class_counts)):
		prob = 0
		for j in range(len(class_counts)):
			if class_counts[j] == i:
				prob += 1
		prob = prob/len(class_counts)
		probs.append(prob)
		probs.append(1-prob)
	return stats.entropy(probs)

print(entropy([0,0,1,1,1,1]))
# print(stats.entropy([2/3,0]))
