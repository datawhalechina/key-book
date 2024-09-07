# 第0章：序言

*编辑：詹好，赵志民*

---

## 关于《机器学习理论导引》

近年来，机器学习领域发展迅猛，相关的课程与教材层出不穷。国内的经典教材如周志华的**《机器学习》**和李航的**《统计学习方法》**，为许多学子提供了机器学习的入门指引。而在国外，Mitchell 的 *Machine Learning*、Duda 等人的 *Pattern Classification*、Alpaydin 的 *Introduction to Machine Learning* 等书籍则提供了更为系统的学习路径。对于希望深入学习的读者，Bishop 的 *Pattern Recognition and Machine Learning*、Murphy 的 *Machine Learning - A Probabilistic Perspective*、Hastie 等人的 *The Elements of Statistical Learning* 等著作也能提供详尽的理论指导。这些书籍无论在国内外，都成为了学习机器学习的重要资源。

然而，从**机器学习理论**的角度来看，现有的学习材料仍存在不足之处。相比于聚焦机器学习算法的著作，专注于机器学习理论的书籍未得到足够的重视。尽管上述一些经典著作中涉及到理论探讨，但篇幅有限，往往仅以独立章节或片段呈现，难以满足深入研究的需求。

以往的机器学习理论经典教材大多为英文撰写。上世纪末围绕统计学习理论展开的讨论，催生了诸如 Vapnik 的 *The Nature of Statistical Learning Theory* 和 *Statistical Learning Theory*，以及 Devroye 等人的 *A Probabilistic Theory of Pattern Recognition* 等经典文献。近年来，Shalev-Shwartz 和 Ben-David 的 *Understanding Machine Learning*，以及 Mohri 等人的 *Foundations of Machine Learning* 进一步推进了这一领域的发展。虽然部分经典著作已有高质量的中文译本，但由中文作者撰写的机器学习理论入门书籍仍显不足。

如今，周志华、王魏、高尉、张利军等老师合著的**《机器学习理论导引》**（以下简称《导引》）填补了这一空白。该书以通俗易懂的语言，为有志于学习和研究机器学习理论的读者提供了良好的入门指引。全书涵盖了**可学性、假设空间复杂度、泛化界、稳定性、一致性、收敛率、遗憾界**七个重要的概念和理论工具。

尽管学习机器学习理论可能不像学习算法那样能够立即应用，但只要持之以恒，深入探究，必将能够领悟到机器学习中的重要思想，并体会其中的深邃奥妙。

-- *詹好*

## 关于《机器学习理论导引》NOTES

《导引》的NOTES在团队内部被亲切地称为《钥匙书》。“钥匙”寓意着帮助读者开启知识之门，解答学习中的疑惑。

《导引》作为一本理论性较强的著作，涵盖了大量数学定理和证明。尽管作者团队已尽力降低学习难度，但由于机器学习理论本身的复杂性，读者仍需具备较高的数学基础。这可能导致部分读者在学习过程中感到困惑，影响学习效果。此外，由于篇幅限制，书中对某些概念和理论的实例说明不足，也增加了理解的难度。

基于以上原因，我们决定编辑这本《钥匙书》作为参考笔记，对《导引》进行深入的注解和补充。其目的是帮助读者更快理解并掌握书中内容，同时记录我们在学习过程中的思考和心得。

《钥匙书》主要包含以下四个部分：

1. **证明补充**：详细解释部分证明的思路，并补充书中省略的证明过程。
2. **案例补充**：增加相关实例，帮助读者加深对抽象概念的理解。
3. **概念补充**：介绍书中涉及但未详细阐释的相关概念。
4. **参考文献讲解**：对部分重要参考文献进行介绍和解读。

鉴于《导引》第一章的内容简明易懂，《钥匙书》从第二章开始详细展开。

对我个人而言，《机器学习理论导引》与*Understanding Machine Learning*和*Foundations of Machine Learning*一样，都是既“无用”又“有用”的书籍。“无用”在于目前的经典机器学习理论尚难全面解释深度学习，尤其是现代生成式大模型的惊人表现。然而，我坚信未来的理论突破将基于现有研究成果，开创新的篇章。因此，分析结论可能并非最重要，真正宝贵的是其中蕴含的思想和分析思路。数学作为一种强有力的工具，能够帮助我们更深入地理解和探索。我期望未来的深度学习能够拥有更多坚实的理论支撑，从而更好地指导实践。正如费曼所言：“What I cannot create, I do not understand.”——“凡我不能创造，我就不能理解。”希望大家能从这些理论中获得启发，创造出更有意义的成果。

另一方面，这本书也让我认识到自身的不足。不同于传统的机器学习算法教材，本书要求读者具备良好的数学功底，通过数学工具从更抽象的角度分析机器学习算法的性质，而非算法本身。学习之路或许漫长，但正如《牧羊少年的奇幻漂流》中所言：“每个人的寻梦过程都是以‘新手的运气’为开端，又总是以‘对远征者的考验’收尾。”希望大家能坚持经历考验，最终实现自己的梦想。

自《钥匙书》v1.0 版本发布以来，受到了众多学习者的关注。我们也收到了许多关于教材内容的疑问。为进一步深入理解相关知识，并记录团队对机器学习理论相关书籍的学习过程，我们将持续对《钥匙书》进行不定期更新，期待大家的关注。

-- *ML67*

## 项目成员

- [王茂霖](https://github.com/mlw67)：第2-7章内容编辑，项目二期更新与修订
- [李一飞](https://github.com/leafy-lee)：第2-7章内容编辑
- [杨昱文](https://github.com/youngfish42)：部分内容编辑
- [谢文睿](https://github.com/Sm1les)：项目激励与支持
- [张雨](https://github.com/Drizzle-Zhang)：第2章部分内容修改
- [J.Hu](https://github.com/inlmouse)：第1章内容编辑
- [赵志民](https://github.com/zhimin-z)：项目二期更新与维护，全书内容编辑
- [詹好](https://github.com/zhanhao93)：项目规划与统筹，全书内容编辑

## 参考文献

Boyd, Stephen, and Lieven Vandenberghe. Convex optimization. Cambridge university press, 2004.

Devroye, Luc, László Györfi, and Gábor Lugosi. A probabilistic theory of pattern recognition. Vol. 31. Springer Science & Business Media, 2013.

Feller, William. "An introduction to probability theory and its applications." (1971).

Kearns, Michael J., and Umesh Vazirani. An introduction to computational learning theory. MIT press, 1994.

McAllester, David A. "PAC-Bayesian stochastic model selection." Machine Learning 51.1 (2003): 5-21.

Mohri, Mehryar. "Foundations of machine learning." (2018).

Wainwright, Martin J. High-dimensional statistics: A non-asymptotic viewpoint. Vol. 48. Cambridge university press, 2019.

Wang, Guanghui, Shiyin Lu, and Lijun Zhang. "Adaptivity and optimality: A universal algorithm for online convex optimization." Uncertainty in Artificial Intelligence. PMLR, 2020.