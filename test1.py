if __name__ == "__main__":
    total = 0
    match = 0
    with open('1czn.sa', 'r') as a, open('1czn_prediction.sa', 'r') as b:
        for aline, bline in zip(a, b):
            for q, w in zip(aline, bline):
                if q == '>' or w == '>':
                    break
                if q == '\n' or w == '\n':
                    continue
                if q == w:
                    match += 1
                total += 1
    mismatch = total - match
    accuracy = match / total
    print("Total: " + str(total))
    print("Correct: " + str(match))
    print("Incorrect: " + str(mismatch))
    print("Percent Correct: " + str(round(accuracy*100, 2)) + "%")
