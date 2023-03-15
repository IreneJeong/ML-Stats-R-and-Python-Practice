####################################################################
#	Illustrate the behaviour of the k-means algorithm for clouds
# of points in the plane. This example is similar to that in 
# Figure 6.2 of  
#
#	Rogers and Girolami (2016), A First Course in Machine Learning,
#	2nd edition, Chapman and Hall/CRC. ISBN: 978-1-49873-8484
#
# mrm: Whalley Range, 6 Feb. 2020
####################################################################
rm( list=ls() ) 
source( "week03_lab_kmeans_steps.r" ) ;

# Read the data, convert into matrix form and look at it
x.df <- read.csv( "rogers_kmeans.csv", header=FALSE ) ;
x.mat <- as.matrix( x.df ) ;
head( x.mat )

# Fix the number of clusters and choose random centres
k.clusters <- 15
crnt.centres <- random.centres( k.clusters, x.mat ) 

# The main event
n.cycles <- 0 ;
max.cycles <- 50 ; # Just so we don't go on forever.
rms.centre.diff <- NA ; # This will get computed before we need it.

while( 
	(n.cycles == 0) || # We've just started
	((rms.centre.diff > 0) && (n.cycles < max.cycles)) # We haven't converged or gone on too long
) {	
	# Assign the points to clusters
	cluster.nums <- assign.to.clusters( x.mat, crnt.centres ) ;
	
	# Find the new centres
	next.centres <- find.centres( k.clusters, cluster.nums, x.mat ) ;
	n.cycles <- n.cycles + 1 ;
	
	# Plot stuff
	# file.name <- paste( "kMeansRG_cycle", n.cycles, ".pdf", sep="" ) ;
	# pdf( file=file.name ) ;
	plot.clusters( x.mat, cluster.nums, next.centres, n.cycles, crnt.centres ) ;
	# dev.off() ;

	if( n.cycles == 1 ) {
		# Plot the initial centres too
		# pdf( file="kMeansRG_start.pdf" ) ;
		plot.clusters( x.mat, cluster.nums, crnt.centres ) ;
		# dev.off()
	}

	# Compute the test for convergence, "rms" is short for
	# "root mean square" and rms.centre diff should be
	# zero if the centres haven't changed.
	dc <- crnt.centres - next.centres ;
	dc.squared <- dc * dc ; # Does elementwise multiplication
	sum.dc.squares <- sum( dc.squared, na.rm=TRUE )
	rms.centre.diff <- sqrt( sum.dc.squares / k.clusters ) ;
	
	# Move on to the next cycle
	crnt.centres <- next.centres ;
}

# Report success, or lack thereof.
if( rms.centre.diff == 0.0 ) {
	# Count the number of nonempty clusters
	n.nonempty <- 0 
	for( j in 1:k.clusters ) {
		if( !any(is.na(crnt.centres[j,])) ) {
			n.nonempty <- n.nonempty + 1
		}
	}
	
	msg <- paste( "Converged with", n.nonempty, "clusters after", n.cycles, "cycles." ) ;
} else {
	msg <- paste( "Failed to converge after", n.cycles, "cycles." ) ;
}

print( msg )
