library(tidyverse)
library(ggmap)

## import sthlm-upps data
setwd('C:/Users/aleen962/Dropbox/PhD/Forskning/Power exponential dist extremes/EpdExtremes/application/data')
load(file = 'wind_gust_data.RData')
load(file = 'wind_dir_data.RData')
load(file = 'wind_gust_list.RData')
load(file = 'wind_dir_list.RData')


## plot sites on map
#
sites <- tibble(lon = unique(wind_gust_data$lon),
               lat = unique(wind_gust_data$lat)
               )
qmplot(x = lon, y = lat, data = sites,
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

### investigate seasonal patterns
# jan-april seem to have the strongest winds
ggplot(wind_gust_data) +
  geom_boxplot(mapping = aes(x = month, y = gust)) +
  geom_hline(yintercept = c(mean(wind_gust_data$gust), 
                            quantile(wind_gust_data$gust, 0.95),
                            quantile(wind_gust_data$gust, 0.99)), 
             color = c('red', 'purple', 'green'), linewidth = 1) +
  labs(x = 'Month',
       y = 'Hourly wind gusts (m/s)') +
  ggtitle('Horizontal lines: mean, 95th and 99th quantiles') +
  theme_bw() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        plot.title = element_text(hjust = 0.5, size = 16))
#
















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







