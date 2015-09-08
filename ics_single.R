#!/usr/bin/env Rscript
library("ICS")
library("mvtnorm")
library("ICSNP")
library("MASS")
args <- commandArgs(trailingOnly = TRUE)

filepath <- paste(args[1], "/gradients.txt", sep="");


X <- read.table(filepath, header=FALSE)
sz <- dim(X)
X <- matrix(data = unlist(lapply(X,as.numeric)),nrow=sz[1], ncol=2)
# X <- matrix(data = sapply(X, as.numeric)), nrow=sz[1])
# X <- matrix(data = as.numeric(unlist(X)), nrow=sz[1])

ics.X <- ics(X, S1 = cov, S2 = tyler.shape )

ics.X.unmix <- coef(ics.X)
X.norm 	    <- X %*% t(ics.X.unmix)



respath <- paste(args[1], "/tmpMat.txt", sep="");
write(ics.X.unmix, respath, 1)

q()
