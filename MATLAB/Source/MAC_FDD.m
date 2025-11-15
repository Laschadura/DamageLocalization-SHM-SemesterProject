function mac=MAC(fai1,fai2)
mac=abs(fai1'*fai2)^2/((fai1'*fai1)*(fai2'*fai2));