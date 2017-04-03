data=read.csv('Boston_Housing.csv')
View(data)
library(caTools)
set.seed(123)
split=sample.split(data,SplitRatio = 0.3)
train=subset(data,split == TRUE)
test=subset(data,split == FALSE)

library(MASS)
reg=glm(formula = MEDV ~ .,data = train)
back = stepAIC(reg,direction = "backward")
y_pred=predict(back,newdata = test)

output=cbind(test$MEDV,y_pred)
colnames(output) = c('test','predicted')
write.csv(output,file='r_output.csv',append = FALSE)

