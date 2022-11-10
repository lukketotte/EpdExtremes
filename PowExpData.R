
setwd("C:/Users/aleen962/Dropbox/PhD/Forskning/Power exponential dist extremes")

library(ncdf4)
dat <- nc_open("tx_ens_mean_0.25deg_reg_2011-2021_v25.0e.nc")
dat

temprature <- ncvar_get(dat,"tx") # very large
longitude <- ncvar_get(dat,"longitude")
latitude <- ncvar_get(dat,"latitude")
times <- ncvar_get(dat,"time")

tail(times)
ncatt_get(dat, "tx", "units")
ncatt_get(dat, 0, "history")

dunits <- ncatt_get(dat, "tx", "units")
tunits <- ncatt_get(dat, "time", "units")


library(chron)
library(lattice)
library(RColorBrewer)

tustr <- strsplit(tunits$value, " ")
tdstr <- strsplit(unlist(tustr)[3], "-")
tmonth <- as.integer(unlist(tdstr)[2])
tday <- as.integer(unlist(tdstr)[3])
tyear <- as.integer(unlist(tdstr)[1])
chron(times, origin=c(tmonth, tday, tyear))

fillvalue <- ncatt_get(dat, "tx", "_FillValue")
temprature[temprature==fillvalue$value] <- NA

length(na.omit(as.vector(temprature[,,1])))
summary(na.omit(as.vector(temprature[,,1])))

m <- 7
summary(na.omit(as.vector(temprature[,,m])))
tmp_slice <- temprature[,,m]

image(longitude,latitude,tmp_slice, col=rev(brewer.pal(10,"RdBu")))

grid <- expand.grid(lon=longitude, lat=latitude)
cutpts <- c(-30,-20,-10,0,10,20,30)
levelplot(tmp_slice ~ longitude * latitude, data=grid, at=cutpts, cuts=7, pretty=T,col.regions=(rev(brewer.pal(10,"RdBu"))))
