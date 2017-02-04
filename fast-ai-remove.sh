#!/bin/bash
aws ec2 disassociate-address --association-id eipassoc-3abdac41
aws ec2 release-address --allocation-id eipalloc-100a2e77
aws ec2 terminate-instances --instance-ids i-075bceb9a4b5e25e1
aws ec2 wait instance-terminated --instance-ids i-075bceb9a4b5e25e1
aws ec2 delete-security-group --group-id sg-8013daf8
aws ec2 disassociate-route-table --association-id rtbassoc-0dac226b
aws ec2 delete-route-table --route-table-id rtb-c4cd8da3
aws ec2 detach-internet-gateway --internet-gateway-id igw-870c48e3 --vpc-id vpc-e81eaa8f
aws ec2 delete-internet-gateway --internet-gateway-id igw-870c48e3
aws ec2 delete-subnet --subnet-id subnet-045b0b5c
aws ec2 delete-vpc --vpc-id vpc-e81eaa8f
echo If you want to delete the key-pair, please do it manually.
