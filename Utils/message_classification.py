class MessageClassification:
    @staticmethod
    def encode(text):
        if text == "ham":
            return 0
        else:
            return 1

    @staticmethod
    def decode(label):
        if label == 0:
            return "ham"
        else:
            return "spam"
