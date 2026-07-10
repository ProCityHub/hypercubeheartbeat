#### STUDY WS19 Mu_Ult2  ####

# Marker: 
# Task 1: Choose fruit
# Marker:
#  58-59 Picture of the person who want's fruit: 58=male 59=female 
#  67-79: Marker M1: Decision+Feedback: 67=correct smile, 68=correct neutral, 69=correct sad, 77=wrong smile,78=wrong neutral, 79=wrong sad  --> M1

# Task 2: Ultimatum as proposer: Only 10 trials, no EEG

# Task 3: Ultimatum as Responder 
#  200-205: Marker M2: Offer; 200: 0 Cent; 201: 1 Cents; 202: 2 Cents; 203: 3 Cents; 204: 4 Cents; 205: 5 Cents --> M2
#  210-222: Marker M3: decision+facial_expression 21_: accept; 22_: reject; 2_0: happy; 2_1: neutral; 2_2: sad  --> M3

rm(list=ls()) # clear workspace
library(ggplot2)
library(reshape2)
library(lmerTest)
library(Rmisc) # Confidence intervalls
library(car)
library(psych)
library(scales)
library(lme4)
library(stats)
library(lavaan)
library(semTools)
library(nlme)

# customized function: calculate standard error
se <- function(x)
{
  y <- x[!is.na(x)] # remove the missing values
  sqrt(var(as.vector(y)))/sqrt(length(y))
}

# customized function: get mean and se
mean_se_aov<-function(form, data, id="") {
  if (id!="") {
    form_2<-deparse(form)
    form_2<-as.formula(paste(form_2,' + ',id))
    data<-aggregate(form_2,data, FUN=mean) }
  end
  df_m<-aggregate(form,data, FUN=mean)
  df_se<-aggregate(form,data, FUN=se)
  df_out<-cbind(df_m,df_se[,dim(df_se)[2]])
  colnames(df_out)[dim(df_out)[2]]<-c("se")
  cp(df_out,col.names=T)
  return(df_out)
}

# customized function: get digit number n from a number x (e.g. "4" is digit #2 from the number 2478)
get_digits <- function(x,n) {
  x<-floor(abs(x))
      mat<-(x %% (10^n)) %/% (10^(n-1))
  return(mat)
}

# customized function: copy
cp <- function(df, sep="\t", dec=",", max.size=(200*1000),row.names=F, col.names=T){
  # Copy a data.frame to clipboard
  write.table(df, paste0("clipboard-", formatC(max.size, format="f", digits=0)), sep=sep, row.names=row.names, dec=dec, col.names=col.names)
}

# Customized function median split
mediansplit<-function(x,as.fac=TRUE) {
    s_med<-median(x[which(!is.na(x))])
      z<-(as.double(x>s_med)) # Mediansplit; Werte auf dem Median werden zu 0

    if (sum(x<s_med,na.rm=TRUE) > sum(x>s_med,na.rm=TRUE))     # Falls es mehr Werte oberhalb as unterhab des Faktors gibt, dann werden Werte auf dem Median als 1 kodiert
      z[x==s_med]<-1
    end
    if (as.fac)
      z<-as.factor(z)
    end
    return(z)
}

# customized function: replace outliers
# for repl='NA', values beyond will be set to NA
# for repl='z', values beyond will be set to the raw value corresponding
# to z standard deviations above or below the mean
# @param ds a data frame
# @param z the standardized criterion, e.g. Z=4, 4 standard deviations above or below the mean
# @param repl what to replace, either NA, or a number in standard deviation
repl_outliers <- function(ds,z=4,repl='NA') {
#  if (!is.data.frame(ds)) {ds<-as.data.frame(ds)}
  zds<-scale(ds)
  if (repl %in% 'NA') {ds[abs(zds)>z]<-NA
  } else {
  mds<-mean(ds,na.rm = TRUE)
  sds<-sd(ds,na.rm = TRUE)
    repl_l=mds-z*sds
    repl_h=mds+z*sds
    ds[ds<repl_l]<-repl_l
    ds[ds>repl_h]<-repl_h
  }
  return(ds)
}


#### M1 - reference task ####

setwd('...') # specify folder with data
ds<-read.csv("M1_single.csv", header=TRUE, sep=",")
ds$decision<-as.numeric(ds$fac_feed<70)              # 1-correct 0-incorrect

ds$face<-get_digits(ds$fac_feed,1) # get last digit
ds$face <- recode(ds$face, "7=1;8=0;9=2")            # 0-neutral 1-smile 2-sad
ds$face<-as.factor(ds$face)
levels(ds$face)<-c('neutral','smile','sad')

ds$face2<-get_digits(ds$fac_feed,1) # get last digit
ds$face2 <- recode(ds$face2, "7=0;8=1;9=2")          # 0-smile 1-neutral 2-sad
ds$face2<-as.factor(ds$face2)
levels(ds$face2)<-c('smile','neutral','sad')
ds$zurtrial<-scale(ds$urtrial)
ds$FRN<-ds$N2-ds$P2

ds_comp<-ds # save as ds_comp for later

# nb of trials
kk<-aggregate(formula=decision~face+VP_nr,data=ds,FUN="length")
describeBy(kk$decision,kk$face)

# An individual-difference indicator for the N2 on smile vs. neutral
ds_M1<-mean_se_aov(N2 ~ face+VP_nr, data = ds, '') # means within VP and condition
ds_M1<-dcast(ds_M1,VP_nr ~ face,value.var='N2') # in wide format
ds_M1$N2_diff_M1<-ds_M1$smile-ds_M1$neutral
hist(ds_M1$N2_diff_M1)
ds_M1<-ds_M1[,c('VP_nr','N2_diff_M1')]
ds_M1$N2_diff_M1<-scale(ds_M1$N2_diff_M1)

# mixed logistic regression
model<-glmer(decision ~ zurtrial + (1 | VP_nr), data = ds, family = binomial,control=glmerControl())
model1<-glmer(decision ~ 1 + (1 | VP_nr), data = ds, family = binomial,control=glmerControl())
summary(model1)
anova(model)
anova(model,model1)

# Plot average number of correct responses per trial
trial_decision<-mean_se_aov(decision ~ urtrial, data = ds, 'VP_nr')
ggplot(trial_decision, aes(urtrial,decision) ) +
  geom_point() +
  stat_smooth()

# Compute confidence interval around the mean number of correct responses
vp_decision<-matrix(,nrow=length(table(ds$VP_nr)), ncol =2)
for (i in unique(ds$VP_nr)) {
 vp_decision[i,1]<-i
 vp_decision[i,2]<-mean(ds[ds$VP_nr==i,'decision'])
}
vp_decision<-as.data.frame(vp_decision)
names(vp_decision)<-c('VP_nr','decision')
hist(vp_decision$decision)
describe(vp_decision$decision) # --> large differences! [.13 .93] 
CI(vp_decision$decision, ci=0.95) # does not include .20 --> They learned something

# psychophysiology

model<-lmer(N170 ~ zurtrial*face + (1 | VP_nr), data = ds)  # Sig. arousal effect neutral>sad,smile
model<-lmer(N2 ~ zurtrial*face + (1 | VP_nr), data = ds)    # Sig, neutral<sad<smile
model<-lmer(P3b ~ zurtrial*face + (1 | VP_nr), data = ds)   # Sig smile<neutral<sad --> Mot. significance, learning from errors

anova(model)
summary(model)



#### Task 2 - Ultimatum game as proposer (no EEG)  ####

setwd('...') # specify folder with data
ds<-read.csv("Task2_export.csv", header=FALSE, sep=",")
names(ds)<-c('VP_nr','urtrial','offer','dec','fac_feed') 
ds[ds$fac_feed<1 |ds$fac_feed>3,"fac_feed"]<-NaN

# Descriptives on offer made and facial expressions sent
ds_offer_T2<-mean_se_aov(offer~VP_nr,data=ds) # offer on average across the 10 trials, per participant 
psych::describe(ds_offer_T2$offer)
hist(ds_offer_T2$offer)

table(ds$dec,ds$fac_feed) # Facial expressions send, according to accept and reject


#### M2 ####
# Task 3, Offer
ds<-read.csv("M2_export.csv", header=TRUE, sep=",")
ds$decision<-ds$decision*(-1)+2 # Recode: 0=reject; 1=accept
ds$FRN<-ds$N2-ds$P2

model<-lmer(N2 ~ condition + (1 | VP_nr), data = ds)    
model<-lmer(P3b ~ condition + (1 | VP_nr), data = ds)   
# Theta: 
ds$condition2<-0
ds[ds$condition=='S205','condition2']<-1
model<-lmer(Theta ~ condition2 + (1 | VP_nr), data = ds) # Theta not sensitive for fair vs. unfair offers!

summary(model)
anova(model)

# Open single trial data for neural responses following the offer 
# (will be used below)
ds<-read.csv("M2_single.csv", header=TRUE, sep=",")
ds$decision <- recode(ds$decision, "1=1;2=0")        # 1-accept 0-reject
ds$foffer <- as.factor(ds$offer)        
ds$decision_n1 <- recode(ds$decision_n1, "1=1;2=0") 
ds$zurtrial<-scale(ds$urtrial)
ds$zoffer<-scale(ds$offer)
ds$zoffer_n1<-scale(ds$offer_n1)
ds$offer_diff<-ds$offer_n1-ds$offer
ds$zoffer_diff<-scale(ds$offer_diff)
ds$FRN<-ds$N2-ds$P2
# scale & outliers
ds$P2<-repl_outliers(ds$P2,z=4,repl='z')
ds$N2<-repl_outliers(ds$N2,z=3,repl='z')
ds$FRN<-repl_outliers(ds$FRN,z=4,repl='z')
ds$P3a<-repl_outliers(ds$P3a,z=4,repl='z')
ds$P3b<-repl_outliers(ds$P3b,z=4,repl='z')
ds$Theta<-repl_outliers(ds$Theta,z=4,repl='z')

ds$zP2<-scale(ds$P2)
ds$zN2<-scale(ds$N2)
ds$zFRN<-scale(ds$FRN)
ds$zP3a<-scale(ds$P3a)
ds$zP3b<-scale(ds$P3b)
ds$zTheta<-scale(ds$Theta)

names(ds)<-paste(names(ds),c("","_M2","",rep("_M2",22)),sep="") # prevent double var-names, except for matching variables
ds_M2<-ds # ds_M2 will be used below

#### M3 ####
# Task 3, facial feedback
ds<-read.csv("M3_single.csv", header=TRUE, sep=",")
ds$decision <- recode(ds$decision, "1=1;2=0")        # 1-accept 0-reject
ds$fdecision <- as.factor(ds$decision)        
levels(ds$fdecision)<-c('reject','accept')
ds$decision_n1 <- recode(ds$decision_n1, "1=1;2=0") 
ds$zurtrial<-scale(ds$urtrial)
ds$zoffer<-scale(ds$offer)
ds$zoffer_n1<-scale(ds$offer_n1)
ds$offer_diff<-ds$offer_n1-ds$offer
ds$offer<-as.factor(ds$offer)
ds$offer_n1<-as.factor(ds$offer_n1)
ds$zoffer_diff<-scale(ds$offer_diff)
ds$face<-get_digits(ds$fac_feed,1) # get last digit  # 0-smile 1-neutral 2-sad
ds$face2<-ds$face # keep this coding for direct comparisons with Task 1
ds$face2<-as.factor(ds$face2)
levels(ds$face2)<-c('smile','neutral','sad')
ds[ds$decision==0,'face'] <- recode(ds[ds$decision==0,'face'], "1=0;2=1")            # 0-more positive (accept-smile & reject-neutral)  -   1-more negative (accept-neutral & reject-sad)
ds$face<-as.factor(ds$face)
levels(ds$face)<-c('more pos','more neg')
ds$fac_feed<-as.factor(ds$fac_feed)
levels(ds$fac_feed)<-c('accept-happy','accept-neutral','reject-neutral','reject-sad')
ds$FRN<-ds$N2-ds$P2
# scale & outliers
ds$N170<-repl_outliers(ds$N170,z=4,repl='z')
ds$P2<-repl_outliers(ds$P2,z=4,repl='z')
ds$N2<-repl_outliers(ds$N2,z=3,repl='z')
ds$FRN<-repl_outliers(ds$FRN,z=4,repl='z')
ds$P3a<-repl_outliers(ds$P3a,z=4,repl='z')
ds$P3b<-repl_outliers(ds$P3b,z=4,repl='z')
ds$Theta<-repl_outliers(ds$Theta,z=4,repl='z')

ds$zN170<-scale(ds$N170)
ds$zP2<-scale(ds$P2)
ds$zN2<-scale(ds$N2)
ds$zFRN<-scale(ds$FRN)
ds$zP3a<-scale(ds$P3a)
ds$zP3b<-scale(ds$P3b)
ds$zTheta<-scale(ds$Theta)

# nb of trials
kk<-aggregate(formula=decision~fac_feed+VP_nr,data=ds,FUN="length")
describeBy(kk$decision,kk$fac_feed)
table(kk[kk$fac_feed=='reject-neutral','decision'])

# Lower compared to higher offers in trial n are followed by lower acceptance rates in trial n
# AV decision n
# UV offer
model<-glmer(decision ~  offer*zurtrial  + (1 | VP_nr), data = ds, family = binomial,control=glmerControl())
summary(model)
anova(model)

# For the overall effect of an independent variable with more than 2 levels, compute the model without the predictor and compare the models  
model1<-glmer(decision ~ zurtrial + offer + (1 | VP_nr), data = ds, family = binomial,control=glmerControl())
anova(model,model1)

# Psychophysiology
# Smiling compared to neutral facial expressions following acceptance elicit a stronger N170 response; a stronger reward positivity (reduced negativity) in the FRN, less theta power, and more positive amplitudes in the P3 component in trial n. 
# Sad compared to neutral facial expressions following rejection in trial n elicit a stronger N170 response; a stronger reward positivity (reduced negativity) in the FRN, less theta power and more positive amplitudes in the P3 component in trial n. 

# AV N170 N2 FRN P3b Theta
# UV dec face
# model<-lmer(Theta ~ fac_feed + (1 | VP_nr), data = ds)  
model<-lmer(N170 ~ fdecision*face + (1 | VP_nr), data = ds) 
model<-lmer(N2 ~ fdecision*face + (1 | VP_nr), data = ds)  
model<-lmer(P3b ~ fdecision*face + (1 | VP_nr), data = ds) 
# Theta for smiling vs neutral (only "accept" trials)
model<-lmer(Theta ~ face + (1 | VP_nr), data = ds[ds$fdecision %in% 'accept',])

cp(anova(model),row.names=T,col.names=F)
summary(model)
anova(model)

# Post hoc
ds1<-ds[ds$fdecision=='reject',] # neutral compared to sad after reject
ds1<-ds[ds$fdecision=='accept',] # smiling compared to neutral after accept
ds1<-ds[(ds$fdecision=='accept' & ds$face == 'more pos') | (ds$fdecision=='reject' & ds$face == 'more neg'),] # smile compared to sad 
model<-lmer(N2 ~ face + (1 | VP_nr), data = ds1)  # Sig. arousal effect neutral>sad,smile

table(ds$face)

# Means for Figure 3A
kk<-mean_se_aov(N2 ~ decision*face*VP_nr, data = ds)
model<-aov(N2 ~ decision*face , data = kk)
mean_se_aov(N2 ~ decision*face, data = kk)


kk<-mean_se_aov(N2 ~ decision*face, data = ds, 'VP_nr')
kk$decision<-as.factor(kk$decision)
kk$decision <- factor(kk$decision, levels = c("1", "0"))
ggplot(kk, aes(decision,N2,fill=face) ) +
  geom_bar(stat="identity", position=position_dodge()) + 
  geom_errorbar(aes(ymin=N2-se, ymax=N2+se), width=.2,
                 position=position_dodge(.9)) + 
  theme(legend.position="none")

# Direct comparison of Task 1 and Task 3, as reported in Supplement A
ds_comp$task<-1 # use saved data from Task 1; insert new variable to indicate the task
ds_comp<-ds_comp[,c("VP_nr","VP_name","face2","zurtrial","task","decision","N170","N2","P3b")] #keep only relevant variables


ds$task<-3 # insert new variable to indicate the task in Task 3

# merge
ds_comp<-rbind(ds_comp,ds[,c("VP_nr","VP_name","face2","zurtrial","task","decision","N170","N2","P3b")]) # merge Task1 and Task3
ds_comp$task<-as.factor(ds_comp$task)

# AV N170 N2 P3b
model<-lmer(N170 ~ face2*zurtrial*task + (1 | VP_nr), data = ds_comp) 
model<-lmer(N2 ~ face2*zurtrial*task + (1 | VP_nr), data = ds_comp)  
model<-lmer(P3b ~ face2*zurtrial*task + (1 | VP_nr), data = ds_comp) 
cp(anova(model),row.names=T,col.names=F)
summary(model)
anova(model)

# Graphs
kk<-mean_se_aov(N170 ~ face2*task,data=ds_comp,id="VP_nr")
kk<-mean_se_aov(N2 ~ face2*task,data=ds_comp,id="VP_nr")
kk<-mean_se_aov(P3b ~ face2*task,data=ds_comp,id="VP_nr")

ggplot(kk, aes(task,P3b,fill=face2) ) +
  geom_bar(stat="identity", position=position_dodge()) + 
  geom_errorbar(aes(ymin=P3b-se, ymax=P3b+se), width=.2,
                 position=position_dodge(.9)) + 
  theme(legend.position="none")

  kk1<-dcast(kk, face ~ decision,value.var='N170')
names(kk1)<-c('face','reject','accept')
ggplot(kk1, aes(face,accept) ) +
  geom_bar(stat="identity", position=position_dodge2()) +
  geom_bar(aes(face,accept),stat="identity", position=position_dodge2(0.9)) 



# Brain behavior
# Higher (compared to lower) acceptance rates in trial n+1 are expected 
# a.	for higher compared to lower offers in trial n+1
# b.	when offers in trial n+1 are higher (rather than lower) compared to offers in trial n (positive reinforcement)
# c.	after smiling compared to neutral facial expression after acceptance in trial n (positive reinforcement) 
# d.	after sad compared to neutral facial expressions after rejection in trial n (successful punishment)
# e.	following a stronger (compared to lower) reward positivity in the FRN, less (compared to more) theta power and more positive (compared to negative) amplitudes in the P3 component after sad facial expression following rejection in trial n (neural indicator of successful punishment) 

# AV dec n+1
# UV offer_n1 (offer_n1-offer) dec face
#    FRN Theta P3b

model<-glmer(decision_n1 ~ offer_n1 + (zurtrial | VP_nr), data = ds, family = binomial,control=glmerControl())
model<-glmer(decision_n1 ~ zoffer + (1  | VP_nr), data = ds, family = binomial,control=glmerControl())
model<-glmer(decision_n1 ~ fdecision*face +  (1 | VP_nr), data = ds, family = binomial,control=glmerControl())
model<-glmer(decision_n1 ~ fdecision*face*zN2 + (zN2 | VP_nr), data = ds, family = binomial,control=glmerControl())
model<-glmer(decision_n1 ~ fdecision*face*zP3b + (zP3b | VP_nr), data = ds, family = binomial,control=glmerControl())

# Posthoc
model<-glmer(decision_n1 ~ face*zP3b + (zP3b | VP_nr), data = ds[ds$decision==1,], family = binomial,control=glmerControl())
model1<-glmer(decision_n1 ~ face+zP3b + (zP3b | VP_nr), data = ds[ds$decision==1,], family = binomial,control=glmerControl())

summary(model)
anova(model)
anova(model,model1)

# graph for decision*face
kk<-mean_se_aov(decision_n1 ~ decision*face, data = ds, 'VP_nr')
kk$decision<-as.factor(kk$decision)
kk$decision <- factor(kk$decision, levels = c("1", "0"))
ggplot(kk, aes(decision,decision_n1,fill=face) ) +
  geom_bar(stat="identity", position=position_dodge()) + 
  geom_errorbar(aes(ymin=decision_n1-se, ymax=decision_n1+se), width=.2,
                 position=position_dodge(.9)) + 
  theme(legend.position="none")+
  scale_y_continuous(limits=c(0.5, 0.62),oob = rescale_none) 

# graph for P3b (w/o VP_nr as random factor)
ggplot(ds, aes(zP3b,decision_n1) ) +
  stat_smooth(method='lm') +
  facet_grid(decision ~ face)

#### Supplemental analyses ####
# as reported in Supplement B
# use neural responses to the offer as predictor of neural responses to the facial expression

ds<-merge(ds,ds_M2,by=c("VP_nr","urtrial"))

# for the post hoc analyses
ds$mN2_M2<-mediansplit(ds$zN2_M2)
ds$mP3b_M2<-mediansplit(ds$zP3b_M2)

# Additional analyses
model1<-lmer(N170 ~ fdecision*face*zN2_M2 + (1 | VP_nr), data = ds)  # Sig. arousal effect neutral>sad,smile
model2<-lmer(N170 ~ fdecision*face*zP3b_M2 + (1 | VP_nr), data = ds)  # Sig. arousal effect neutral>sad,smile
model3<-lmer(N2   ~ fdecision*face*zN2_M2 + (1 | VP_nr), data = ds)    # Sig, neutral<sad<smile
model4<-lmer(N2   ~ fdecision*face*zP3b_M2 + (1 | VP_nr), data = ds)    # Sig, neutral<sad<smile
model5<-lmer(P3b  ~ fdecision*face*zN2_M2 + (1 | VP_nr), data = ds)   # Sig smile<neutral<sad --> Mot. significance, learning from errors
model6<-lmer(P3b  ~ fdecision*face*zP3b_M2 + (1 | VP_nr), data = ds)   # Sig smile<neutral<sad --> Mot. significance, learning from errors
model6a<-lmer(P3b  ~ fdecision*face + (1 | VP_nr), data = ds[ds$mP3b_M2==1,])   # Sig smile<neutral<sad --> Mot. significance, learning from errors

#Post hoc
model3a<-lmer(N2   ~ zN2_M2 + (1 | VP_nr), data = ds[ds$offer_M2==0 & ds$fdecision=="accept" & ds$face=="more pos",])    # Sig, neutral<sad<smile
model3b<-lmer(N2   ~ zN2_M2 + (1 | VP_nr), data = ds[ds$offer_M2==1 & ds$fdecision=="accept" & ds$face=="more pos",])    # Sig, neutral<sad<smile
model3c<-lmer(N2   ~ zN2_M2 + (1 | VP_nr), data = ds[ds$offer_M2==2 & ds$fdecision=="accept" & ds$face=="more pos",])    # Sig, neutral<sad<smile
model3d<-lmer(N2   ~ zN2_M2 + (1 | VP_nr), data = ds[ds$offer_M2==3 & ds$fdecision=="accept" & ds$face=="more pos",])    # Sig, neutral<sad<smile
model3e<-lmer(N2   ~ zN2_M2 + (1 | VP_nr), data = ds[ds$offer_M2==4 & ds$fdecision=="accept" & ds$face=="more pos",])    # Sig, neutral<sad<smile
model3f<-lmer(N2   ~ zN2_M2 + (1 | VP_nr), data = ds[ds$offer_M2==5 & ds$fdecision=="accept" & ds$face=="more pos",])    # Sig, neutral<sad<smile

model3g<-lmer(N2   ~ zN2_M2 + (1 | VP_nr), data = ds[ds$fdecision=="reject" & ds$face=="more neg",])    # Sig, neutral<sad<smile

kk<-anova(model,model1)
anova(model6a)
summary(model)

# graphs
kk<-mean_se_aov(N2 ~ fdecision*face,data=ds[ds$mN2_M2==0,],id="VP_nr") # offer_M2*N2_M2 on N2
kk<-mean_se_aov(P3b ~ fdecision*face,data=ds[ds$mP3b_M2==0,],id="VP_nr") 

ggplot(kk, aes(fdecision,N2,fill=face) ) +
  geom_bar(stat="identity", position=position_dodge()) + 
  geom_errorbar(aes(ymin=N2-se, ymax=N2+se), width=.2,
                 position=position_dodge(.9)) 
ggplot(kk, aes(fdecision,P3b,fill=face) ) +
  geom_bar(stat="identity", position=position_dodge()) + 
  geom_errorbar(aes(ymin=P3b-se, ymax=P3b+se), width=.2,
                 position=position_dodge(.9)) + 
  ylim(0,2.5)


table(ds$mN2_M2)

ds$cdecision<-as.numeric(ds$decision)
kk<-mean_se_aov(cdecision ~ VP_nr*mP3b_M2,data=ds[ds$offer_M2==0,]) 

hist(kk[kk$mP3b_M2==1,"cdecision"])
