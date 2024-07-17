import jieba


class JiebaModel:
    def load_model(self):
        self.jieba_model = jieba.lcut

    def generate_result(self, text):
        return self.jieba_model(text, cut_all=False)


j = JiebaModel()
j.load_model()
print(j.generate_result("我很好我很强大"))