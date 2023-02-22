####################################################################
#	Implement the steps in the k-means algorithm as sketched
# in the lecture on clustering from Stats & Machine Learning 2.
#
# mrm: Whalley Range, 6 Feb. 2020
####################################################################

 
##################################################################
# Step 1: Choose initial values for the centres
#
# There are *lots* of ways to do this. Here we define one function
# that, given k and the data, chooses centres uniformly at random
# from the region containing the data. The data should be in a
# matrix, with one point per row.
##################################################################

random.centres <- function( n.clusters, x.mat )
{
	# Work out the dimension of the data,
	# then get the range of each coordinate
	x.dim <- ncol( x.mat )
	x.min <- rep( NA, x.dim ) ;
	x.max <- rep( NA, x.dim ) ;
	for( j in 1:x.dim ) {
		x.min[j] <- min( x.mat[,j] ) ;
		x.max[j] <- max( x.mat[,j] ) ;
	}
	
	# Initialise an empty matrix of centres, then fill it
	# by choosing coords uniformly at random from the data's range.
	centre.mat <- matrix( rep( NA, n.clusters*x.dim ), nrow=n.clusters ) ;
	for( j in 1:x.dim ) {
		centre.mat[,j] <- runif( n.clusters, min=x.min[j], max=x.max[j] ) ;
	}
	
	return( centre.mat ) ;
}

# Choose actual data points as centres
centres.from.data <- function( n.clusters, x.mat )
{
	# Examine the inputs to get the dimension
	# of the data and the number of points.
	x.dim <- ncol( x.mat )
	n.pts <- nrow( x.mat )
	
	# Initialise an empty matrix of centres, then fill it
	# by choosing coords uniformly at random from the data's range.
	centre.mat <- matrix( rep( NA, n.clusters*x.dim ), nrow=n.clusters ) ;
	chosen.pt.nums <- sample.int( n.pts, size=n.clusters, replace=FALSE  ) ;
	for( j in 1:n.clusters ) {
		centre.mat[j,] <- x.mat[chosen.pt.nums[j],]
	}
	
	return( centre.mat ) ;
}

##################################################################
# Step 2: Assign the points to clusters
#
# For each point, compute the distances to all the centres 
# and assign the point to that cluster whose centre is nearest.
##################################################################

assign.to.clusters <- function( x.mat, centre.mat )
{
	# Look at the data to get various useful numbers
	n.pts <- nrow( x.mat ) ;
	x.dim <- ncol( x.mat ) ;
	n.clusters <- nrow( centre.mat ) ;
	
	# Initialise the result, which is a vector that lists the
	# cluster assignments. We set it to NA so that if, somehow,
	# we fail to assign a point to a cluster, the problem will be obvious.
	cluster.for.x <- rep( NA, n.pts ) ;
	
	# Now classify the points
	crnt.dists <- rep( NA, n.clusters ) ; # We'll use this repeatedly
	for( i in 1:n.pts ) {
		# Compute the distances
		crnt.x <- x.mat[i,] ;
		for( j in 1:n.clusters ) {
			# Compute the distance to the centre of the j-th cluster
			if( any(is.na(centre.mat[j,])) ) {
				# The j-th cluster is empty
				crnt.dists[j] <- NA ;
			} else {
				# The j-th centre contains some points
				dx <- crnt.x - centre.mat[j,] ;
				dx.squared <- dx * dx ; # element-by-element multiplication
				crnt.dists[j] <- sqrt( sum(dx.squared) ) ;
			}
		}
		
		# Find the number of the cluster that's closest
		# using R's built-in function which.min(), which
		# finds the position in a list at which the list's
		# minimum value appears. If the minimum appears more
		# than once, which.min() returns the position of the
		# first appearance. NA's get ignored.
		cluster.for.x[i] <- which.min( crnt.dists ) ;
	}
	
	return( cluster.for.x ) ;
}

##################################################################
# Step 3: Recompute the centres based on the cluster assignments
##################################################################

find.centres <- function( n.clusters, cluster.for.x, x.mat ) 
{
	# Look at the data to get various useful numbers
	n.pts <- nrow( x.mat ) ;
	x.dim <- ncol( x.mat ) ;
	
	# Run through the points accumulating sums of the
	# vectors in the clusters and counting the number of
	# points in each cluster. We'll store the sum in a
	# matrix with one row per sum (or, equivalently, one
	# row per cluster).
	pts.in.cluster <- rep( 0, n.clusters ) ;
	sum.x.in.cluster <- matrix( rep(0, n.clusters*x.dim), nrow=n.clusters ) ;
	for( j in 1:n.pts ) {
		ncx <- cluster.for.x[j] ; # Number of the cluster containing x
		pts.in.cluster[ncx] <- 1 + pts.in.cluster[ncx] ;
		sum.x.in.cluster[ncx,] <- sum.x.in.cluster[ncx,] + x.mat[j,]
	}
	
	# Now divide the sums by the relevant numbers of points
	# to get the new centre positions.
	for( j in 1:n.clusters ) {
		# Check for empty clusterss
		if( pts.in.cluster[j] == 0 ) {
			# Note that the cluster is empty by
			# setting the coords of its centre to NA.
			sum.x.in.cluster[j,] <- rep( NA, x.dim ) ;
		} else {
			# The cluster contains points, so we should 
			# get the centre's position by computing a mean
			sum.x.in.cluster[j,] <- sum.x.in.cluster[j,] / pts.in.cluster[j] ;
		}
	}
	
	# Now that we've normalised the sums, they're
	# means over x's in the clusters, so we're finished.
	return( sum.x.in.cluster ) ;
}

##################################################################
#	Given centres and data, compute the within-cluster 
# sum-of-squares
##################################################################

sum.of.squares <- function( centre.mat, x.mat, cluster.for.x )
{
	# Examine the inputs to get the number clusters,
	# the dimension of the data and the number of points.
	n.pts <- nrow( x.mat ) ;
	x.dim <- ncol( x.mat ) ;
	k.clusters <- nrow( centre.mat ) ;
	
	sum.o.squares <- 0.0 ;
	for( j in 1:n.pts ) {
		ncx <- cluster.for.x[j] ; # ncx: number of cluster containing x
		dx <- x.mat[j,] - centre.mat[ncx,] ;
		dx.squared <- dx*dx ; # does elementwise multiplication
		sum.o.squares <- sum.o.squares + sum( dx.squared ) ;
	}

	return( sum.o.squares ) ;
}
##################################################################
# Define a function to plot the points and centres
##################################################################

library( RColorBrewer )
pair.pal <- brewer.pal( 12, "Paired" ) ; # Get some pleasing colours
light.pal <- pair.pal[c(1,3,5,7,9,11)] ;
dark.pal <- pair.pal[c(2,4,6,8,10,12)] ;

plot.clusters <- function( 
	x.mat, cluster.for.x, crnt.centres, n.cycles=NULL, old.centres=NULL, my.cex=1.0
) 
{
	# First plot all the points. If there is an iteration
	# number, use it in the plot.
	title.str <- "Initial clusters" ;
	if( !is.null(n.cycles) ) {
		if( n.cycles == 1 ) {
			title.str <- "After the first cycle" ;
		} else {
			title.str <- paste( "After", n.cycles, "cycles" ) ;
		}
	}
		
	plot( x.mat[,1], x.mat[,2], main=title.str,
		xlab=expression( "x"[1] ),
		ylab=expression( "x"[2] ),
		type="p", pch=19, cex=my.cex, col=dark.pal[cluster.for.x]
	) ;
			
	# If old centres are available show them,
	# along with dashed lines to the new centres
	if( !is.null(old.centres) ) {
		# Plot filled diamonds for the old centres
		points( old.centres[,1], old.centres[,2], 
			pch=23, cex=1.5*my.cex, col="black", bg=dark.pal
		) ;
		
		k.centres <- nrow( crnt.centres ) ;
		for( k in 1:k.centres ) {
			lines( 
				x=c(old.centres[k,1], crnt.centres[k,1]),
				y=c(old.centres[k,2], crnt.centres[k,2]),
				col="black", lty="dashed"
			) ;
		}
	}
	
	if( !is.null( n.cycles ) ) {
		# Add diamonds for the current centres
		points( crnt.centres[,1], crnt.centres[,2], 
			pch=23, cex=1.5*my.cex, col="black", bg=dark.pal 
		) ;
	}
}