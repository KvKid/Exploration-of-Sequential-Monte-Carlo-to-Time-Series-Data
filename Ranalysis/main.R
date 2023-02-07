install.packages("xts", repos="http://cloud.r-project.org")
install.packages("forecast")
library("xts")
library("forecast")

data <- read.csv("./processeddata.csv")

data$Date <-as.Date(data$Date)

ts <- xts(data$a21, data$Date)
attr(ts, 'frequency') <- 7 


plot(ts,ylab = 'Freq')

plot(aggregate())

ts.decomp <- decompose(as.ts(ts), type = 'additive')

plot(ts.decomp)
#Variance remains fairly constant for randomness so choose additive instead of multiplicative
# (Choose multiplicative if the variance increases)


stl(as.ts(ts))

plot(aggregate(as.ts(ts)))

ts.diff <-diff(as.ts(ts),lag = 5)

ts.diff.decomp <- decompose(ts.diff, type = "additive")
plot(ts.diff.decomp)

auto.arima(ts.diff,max.p = 4, max.q = 4)
acf(ts.diff)
pacf(ts.diff)
