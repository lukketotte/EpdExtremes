rm(list = ls())
dev.off()
gc()
# .rs.restartR()

library(ggmap)
library(raster)
library(padr)
library(anytime)
library(tidyverse)

## import sthlm-upps data
setwd('C:/Users/aleen962/Dropbox/PhD/Forskning/Power exponential dist extremes/EpdExtremes/application/data')

wind_gust_coordinates <- read_csv(file = 'wind_gust_coordinates.csv')
wind_gust_coordinates_km <- read_csv(file = 'wind_gust_coordinates_km.csv')
wind_gust_coordinates_unit <- read_csv(file = 'wind_gust_coordinates_unit.csv')

load(file = 'wind_gust_data_full.RData')
load(file = 'wind_dir_data.RData')

load(file = 'wind_gust_list.RData')
load(file = 'wind_dir_list.RData')


############################ ############################
### exploratory analysis ### ### exploratory analysis ###
############################ ############################

## plot sites on map
#
qmplot(x = lon, y = lat, data = wind_gust_coordinates,
       maptype = "toner-lite",
       extent = "normal",
       xlab = 'Longitude',
       ylab = 'Latitude',
       f = 0.2,
       zoom = 9,
       colour = I('red'),
       size = I(3),
       shape = I(8),
       stroke = I(1.5)
       ) + 
  theme_bw() +
  theme(legend.position = "none",
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()
        ) 

# maptype = c("terrain", "terrain-background", "toner", "watercolor", "terrain-labels", 
#             "terrain-lines", "toner-2010", "toner-2011", "toner-background", 
#             "toner-hybrid", "toner-labels", "toner-lines", "toner-lite")
#


### investigate seasonal patterns with box-plots
# jan-april seem to have the strongest winds
ggplot(wind_gust_data_full) +
  geom_boxplot(mapping = aes(x = month, y = gust)) +
  geom_hline(yintercept = c(mean(wind_gust_data_full$gust, na.rm = TRUE), 
                            quantile(wind_gust_data_full$gust, 0.95, na.rm = TRUE),
                            quantile(wind_gust_data_full$gust, 0.99, na.rm = TRUE)), 
             color = c('red', 'purple', 'green'), linewidth = 1) +
  labs(x = 'Month',
       y = 'Hourly wind gusts (m/s)') +
  ggtitle('Horizontal lines: mean, 95th and 99th quantiles') +
  theme_bw() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        plot.title = element_text(hjust = 0.5, size = 16))
#

### print number of obs per site
stations <- wind_gust_data_full %>% distinct(station)
for(i in 1:nrow(stations)){
  wind_gust_data_full %>% 
  filter(station == stations$station[i]) %>% 
  nrow() %>% 
  print()
}
#

### pad missing values
wind_gust_data_full <- wind_gust_data_full %>%
  mutate(date_full = with(wind_gust_data_full, anytime::anytime(paste(date, time)))) %>%
  group_by(station) %>%
  padr::pad(interval = 'hour', by = 'date_full') %>% #, start_val = as.POSIXlt('2019-01-01'), end_val = as.POSIXlt('2022-12-31')) %>%
  ungroup()
#

### extract months with strongest wind gusts
season <- c('Jan', 'Feb', 'Mar', 'Apr')

wind_gust_season <- wind_gust_data_full %>% 
  filter(month %in% season)

wind_gust_season
#
nrow(wind_gust_season) / nrow(distinct(wind_gust_season, station))
#
for(i in 1:nrow(stations)){
  wind_gust_season %>% 
  filter(station == stations$station[i]) %>% 
  nrow() %>% 
  print()
}
#

ggplot(wind_gust_season) + 
  geom_point(aes(date, gust)) +
  theme_bw()
#
wind_gust_season %>% distinct(month)
#
summary(wind_gust_season)
#
ggplot(wind_gust_season) +
  geom_boxplot(mapping = aes(x = month, y = gust)) +
  geom_hline(yintercept = c(mean(wind_gust_season$gust), 
                            quantile(wind_gust_season$gust, 0.95),
                            quantile(wind_gust_season$gust, 0.99)), 
             color = c('red', 'purple', 'green'), linewidth = 1) +
  labs(x = 'Month',
       y = 'Hourly wind gusts (m/s)') +
  ggtitle('Horizontal lines: mean, 95th and 99th quantiles') +
  theme_bw() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        plot.title = element_text(hjust = 0.5, size = 16))
#


stats <- stations$station
nas <- numeric()
full_dat <- NULL
for(j in 1:length(stats)){
  test2 <- NULL
  for(i in 1:4){
    test_dat <- wind_gust_season %>% 
      filter(year == 2018+i & station == stations$station[j])
    test <- test_dat %>% 
      mutate(date_full = with(test_dat, anytime(paste(date, time)))) %>% 
      padr::pad(interval = 'hour', by = 'date_full')
    test2 <- rbind(test2, test)
    }
  nas[j] <- test2 %>% 
    mutate(na_ratio = mean(is.na(gust))) %>% 
    distinct(na_ratio)
  full_dat <- rbind(full_dat, test2)
}
na_ratios <- unlist(nas)
summary(na_ratios)

full_dat %>% 
  mutate(na_ratio = mean(is.na(gust))) %>% 
  distinct(na_ratio)

test <- test_dat %>% 
  mutate(date_full = with(test_dat, anytime(paste(date, time)))) %>% 
  padr::pad(interval = 'hour', by = 'date_full') %>% 
  mutate(na_ratio = mean(is.na(gust))) %>% 
  print()

tidyr::complete(test_dat, time = seq(min(time), max(time), by = "1 hour"))

test <- wind_gust_season %>% 
  mutate(date_full = with(wind_gust_season, anytime(paste(date, time)))) %>% 
  group_by(station) %>% 
  padr::pad(interval = 'hour', by = 'date_full') %>% 
  ungroup()

#
for(i in 1:nrow(stations)){
  test %>% 
  filter(station == stations$station[i]) %>% 
  nrow() %>% 
  print()
}

test %>% 
  group_by(station) %>% 
  mutate(na_ratio = mean(is.na(gust))) %>% 
  ungroup() %>% 
  distinct(na_ratio)

test2 <- NULL
for(i in 1:4){
  test2 <- bind_rows(test2, test %>% 
                       dplyr::filter(year == 2018+i) %>% 
                       dplyr::select(gust, station) #%>% 
                     # tidyr::pivot_wider(names_from = station, values_from = gust)
  )
}
test2


stations <- wind_gust_season %>% distinct(station)
stations$station[1]

for(i in 1:nrow(stations)){
  wind_gust_season %>% 
  filter(station == stations$station[i]) %>% 
  nrow() %>% 
  print()
}

wind_gust_season %>% 
  filter(station == stations$station[2]) %>% 
  summary()


# save data as csv
setwd('C:/Users/aleen962/Dropbox/PhD/Forskning/Power exponential dist extremes/EpdExtremes/application/data')

swe_wind_data <- NULL
for(i in 1:4){
  swe_wind_data <- bind_rows(swe_wind_data, wind_gust_season %>% 
                            dplyr::filter(year == 2018+i) %>% 
                            dplyr::select(gust, station) %>% 
                            tidyr::pivot_wider(names_from = station, values_from = gust)
  )
}

test <- wind_gust_season %>% 
  dplyr::filter(year == 2018+1) %>% 
  dplyr::select(gust, station)

test %>% pivot_wider(names_from = 'station', values_from = 'gust')


print(swe_wind_data, n=50)
#
cal_temp_coordinates <- wind_gust_season %>% 
  distinct(lon, lat)
#



















### calculate extremal coefficients

# transform data to matrices
model_data <- as.matrix(wind_gust_data_full)
coords <- as.matrix(wind_gust_coordinates_km)

# estimate marginals
i <- 1 + i
est <- extRemes::fevd(model_data[,i], type = "GEV", method = "MLE", time.units = "years", period.basis = "year")
plot(est)
summary(est)

# transform data to unit fréchet
frech_fun <- function(data){
  res <- extRemes::fevd(data, type = "GEV", method = "MLE", time.units = "years", period.basis = "year")
  pars <- res$results$par
  
  return(SpatialExtremes::gev2frech(data, loc = pars[1], scale = pars[2], shape = pars[3]))
}
dat <- apply(X = model_data, MARGIN = 2, FUN = frech_fun)

# fit Brown-Resnick model to data
mod <- SpatialExtremes::fitmaxstab(data = dat,
                                   coord = as.matrix(wind_gust_coordinates_km),
                                   cov.mod = "brown")
mod$param

# calculate extremal coefficients
extremal_coefficients <- SpatialExtremes::fitextcoeff(data = dat, coord = coords, 
                                                      estim = "ST", marge = "frech", 
                                                      plot = TRUE, loess = TRUE, 
                                                      method = "BFGS", std.err = TRUE,
                                                      prob = 0
                                                      )
extremal_coefficients

extremal_coeff <- tibble(distance = extremal_coefficients$ext.coeff[,1],
                         ext_coeff = extremal_coefficients$ext.coeff[,2])

summary(extremal_coefficients$ext.coeff[,1])


# plot extremal coefficients against distance
ext_coeff_fig <- ggplot(data = extremal_coeff, aes(x = distance, y = ext_coeff)) + 
  geom_point(shape = 1) + 
  geom_smooth(se = FALSE, color = 'black') +
  labs(x = 'Distance (km)',
       # y = expression( paste(hat(theta), '(h)', sep = ''))) +
       y = expression( theta)) +
  ggtitle('Estimated extremal coefficients') +
  theme_bw(base_size = 15) +
  theme(plot.title = element_blank(),#  element_text(hjust = 0.5),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()
        # axis.title.y = element_text(angle = 0, vjust = 0.5)
        )
ext_coeff_fig















######################################################################################
############################## OKLAHOMA WIDF GUST DATA ###############################
######################################################################################

### import data
wind_dat <- read_csv(file = "Data/Oklahoma_wind_gust_data.csv", col_names = TRUE)
wind_sites <- read_csv(file = "Data/Oklahoma_geo_info.csv", col_names = TRUE)

### extract relevant variables and rename
wind_dat <- wind_dat %>% 
  mutate('date' = make_date(year = YEAR, month = MONTH, day = DAY)) %>% 
  transmute(Date = date, Site.Id = STID, Wind.max = WMAX)
wind_dat$Wind.max[wind_dat$Wind.max < 0] <- NA
wind_dat

wind_sites <- wind_sites %>% 
  transmute(Site.Id = stid, long = elon, lat = nlat)

### join data sets
wind_dat <- left_join(wind_dat, wind_sites, by = "Site.Id")
dim(wind_dat)

### remove sites with many missing values
wind_dat %>% 
  group_by(Site.Id) %>% 
  summarise(Num.na = sum(is.na(Wind.max))) %>% 
  print(n = 150)

wind_dat <- wind_dat %>% 
  group_by(Site.Id) %>% 
  mutate(Num.na = sum(is.na(Wind.max))) %>% 
  ungroup()

wind_dat <- wind_dat[wind_dat$Num.na < 100, ]
length(unique(wind_dat$Site.Id))

### create map with sites
library(ggmap)
state = map_data("state")
oklahoma_coords <- state[state$region == 'oklahoma', ][ , c(1, 2)]

qmplot(x = long, y = lat, data = wind_dat, 
       maptype = "terrain", 
       extent = "panel",
       xlab = 'Longitude',
       ylab = 'Latitude',
       f = 0.1
       ) 
#

maptype = c("terrain", "terrain-background", "satellite", "roadmap", "hybrid", "toner",
            "watercolor", "terrain-labels", "terrain-lines", "toner-2010", "toner-2011",
            "toner-background", "toner-hybrid", "toner-labels", "toner-lines", "toner-lite")
#
# end Oklahoma wind gust data








########## map stuff

# only violent crimes
violent_crimes <- subset(crime,
  offense != "auto theft" &
  offense != "theft" &
  offense != "burglary"
)

# rank violent crimes
violent_crimes$offense <- factor(
  violent_crimes$offense,
  levels = c("robbery", "aggravated assault", "rape", "murder")
)

# restrict to downtown
violent_crimes <- subset(violent_crimes,
  -95.39681 <= lon & lon <= -95.34188 &
   29.73631 <= lat & lat <=  29.78400
)

theme_set(theme_bw())

qmplot(lon, lat, data = violent_crimes, colour = offense,
  size = I(3.5), alpha = I(.6), legend = "topleft")

qmplot(lon, lat, data = violent_crimes, geom = c("point","density2d"))
qmplot(lon, lat, data = violent_crimes) + facet_wrap(~ offense)
qmplot(lon, lat, data = violent_crimes, extent = "panel") + facet_wrap(~ offense)
qmplot(lon, lat, data = violent_crimes, extent = "panel", colour = offense, darken = .4) +
  facet_wrap(~ month)




qmplot(long, lat, xend = long + delta_long,
  color = I("red"), yend = lat + delta_lat, data = seals,
  geom = "segment", zoom = 5)

qmplot(long, lat, xend = long + delta_long, maptype = "watercolor",
  yend = lat + delta_lat, data = seals,
  geom = "segment", zoom = 6)

qmplot(long, lat, xend = long + delta_long, maptype = "terrain",
  yend = lat + delta_lat, data = seals,
  geom = "segment", zoom = 6)


qmplot(lon, lat, data = wind, size = I(.5), alpha = I(.5)) +
  ggtitle("NOAA Wind Report Sites")

# thin down data set...
s <- seq(1, 227, 8)
thinwind <- subset(wind,
  lon %in% unique(wind$lon)[s] &
  lat %in% unique(wind$lat)[s]
)

# for some reason adding arrows to the following plot bugs
theme_set(theme_bw(18))

qmplot(lon, lat, data = thinwind, geom = "tile", fill = spd, alpha = spd,
    legend = "bottomleft") +
  geom_leg(aes(xend = lon + delta_lon, yend = lat + delta_lat)) +
  scale_fill_gradient2("Wind Speed\nand\nDirection",
    low = "green", mid = scales::muted("green"), high = "red") +
  scale_alpha("Wind Speed\nand\nDirection", range = c(.1, .75)) +
  guides(fill = guide_legend(), alpha = guide_legend())




## kriging
############################################################
# the below examples show kriging based on undeclared packages
# to better comply with CRAN's standards, we remove it from
# executing, but leave the code as a kind of case-study
# they also require the rgdal library


library(lattice)
library(sp)
library(rgdal)

# load in and format the meuse dataset (see bivand, pebesma, and gomez-rubio)
data(meuse)
coordinates(meuse) <- c("x", "y")
proj4string(meuse) <- CRS("+init=epsg:28992")
meuse <- spTransform(meuse, CRS("+proj=longlat +datum=WGS84"))

# plot
plot(meuse)

m <- data.frame(slot(meuse, "coords"), slot(meuse, "data"))
names(m)[1:2] <- c("lon", "lat")

qmplot(lon, lat, data = m)
qmplot(lon, lat, data = m, zoom = 14)


qmplot(lon, lat, data = m, size = zinc,
  zoom = 14, source = "google", maptype = "satellite",
  alpha = I(.75), color = I("green"),
  legend = "topleft", darken = .2
) + scale_size("Zinc (ppm)")








# load in the meuse.grid dataset (looking toward kriging)
library(gstat)
data(meuse.grid)
coordinates(meuse.grid) <- c("x", "y")
proj4string(meuse.grid) <- CRS("+init=epsg:28992")
meuse.grid <- spTransform(meuse.grid, CRS("+proj=longlat +datum=WGS84"))

# plot it
plot(meuse.grid)

mg <- data.frame(slot(meuse.grid, "coords"), slot(meuse.grid, "data"))
names(mg)[1:2] <- c("lon", "lat")

qmplot(lon, lat, data = mg, shape = I(15), zoom = 14, legend = "topleft") +
  geom_point(aes(size = zinc), data = m, color = "green") +
  scale_size("Zinc (ppm)")



# interpolate at unobserved locations (i.e. at meuse.grid points)
# pre-define scale for consistency
scale <- scale_color_gradient("Predicted\nZinc (ppm)",
  low = "green", high = "red", lim = c(100, 1850)
)



# inverse distance weighting
idw <- idw(log(zinc) ~ 1, meuse, meuse.grid, idp = 2.5)
mg$idw <- exp(slot(idw, "data")$var1.pred)

qmplot(lon, lat, data = mg, shape = I(15), color = idw,
  zoom = 14, legend = "topleft", alpha = I(.75), darken = .4
) + scale



# linear regression
lin <- krige(log(zinc) ~ 1, meuse, meuse.grid, degree = 1)
mg$lin <- exp(slot(lin, "data")$var1.pred)

qmplot(lon, lat, data = mg, shape = I(15), color = lin,
  zoom = 14, legend = "topleft", alpha = I(.75), darken = .4
) + scale



# trend surface analysis
tsa <- krige(log(zinc) ~ 1, meuse, meuse.grid, degree = 2)
mg$tsa <- exp(slot(tsa, "data")$var1.pred)

qmplot(lon, lat, data = mg, shape = I(15), color = tsa,
  zoom = 14, legend = "topleft", alpha = I(.75), darken = .4
) + scale



# ordinary kriging
vgram <- variogram(log(zinc) ~ 1, meuse)   # plot(vgram)
vgramFit <- fit.variogram(vgram, vgm(1, "Exp", .2, .1))
ordKrige <- krige(log(zinc) ~ 1, meuse, meuse.grid, vgramFit)
mg$ordKrige <- exp(slot(ordKrige, "data")$var1.pred)

qmplot(lon, lat, data = mg, shape = I(15), color = ordKrige,
  zoom = 14, legend = "topleft", alpha = I(.75), darken = .4
) + scale



# universal kriging
vgram <- variogram(log(zinc) ~ 1, meuse) # plot(vgram)
vgramFit <- fit.variogram(vgram, vgm(1, "Exp", .2, .1))
univKrige <- krige(log(zinc) ~ sqrt(dist), meuse, meuse.grid, vgramFit)
mg$univKrige <- exp(slot(univKrige, "data")$var1.pred)

qmplot(lon, lat, data = mg, shape = I(15), color = univKrige,
  zoom = 14, legend = "topleft", alpha = I(.75), darken = .4
) + scale



# adding observed data layer
qmplot(lon, lat, data = mg, shape = I(15), color = univKrige,
  zoom = 14, legend = "topleft", alpha = I(.75), darken = .4
) +
  geom_point(
    aes(x = lon, y = lat, size = zinc),
    data = m, shape = 1, color = "black"
  ) +
  scale +
  scale_size("Observed\nLog Zinc")







