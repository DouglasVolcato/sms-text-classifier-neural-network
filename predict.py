from Utils.text_classifier_model import TextClassifierModel

model = TextClassifierModel()

print(f"{model.predict("07732584351 - Rodger Burns - MSG = We tried to call you re your reply to our sms for a free nokia mobile + free camcorder. Please call now 08000930705 for delivery tomorrow")} should be spam")
print(f"{model.predict("Someone U know has asked our dating service 2 contact you! Cant Guess who? CALL 09058097189 NOW all will be revealed. POBox 6, LS15HB 150p")} should be spam")
print(f"{model.predict("Hottest pics straight to your phone!! See me getting Wet and Wanting, just for you xx Text PICS to 89555 now! txt costs 150p textoperator g696ga 18 XxX")} should be spam")
print(f"{model.predict("Had your contract mobile 11 Mnths? Latest Motorola, Nokia etc. all FREE! Double Mins & Text on Orange tariffs. TEXT YES for callback, no to remove from records.")} should be spam")

print(f"{model.predict("I got another job! The one at the hospital doing data analysis or something, starts on monday! Not sure when my thesis will got finished")} should be ham")
print(f"{model.predict("Haven't eaten all day. I'm sitting here staring at this juicy pizza and I can't eat it. These meds are ruining my life")} should be ham")
print(f"{model.predict("Should i buy him a blackberry bold 2 or torch. Should i buy him new or used. Let me know. Plus are you saying i should buy the  &lt;#&gt; g wifi ipad. And what are you saying about the about the  &lt;#&gt; g?")} should be ham")
print(f"{model.predict("Do whatever you want. You know what the rules are. We had a talk earlier this week about what had to start happening, you showing responsibility. Yet, every week it's can i bend the rule this way? What about that way? Do whatever. I'm tired of having thia same argument with you every week. And a  &lt;#&gt;  movie DOESNT inlude the previews. You're still getting in after 1.")} should be ham")