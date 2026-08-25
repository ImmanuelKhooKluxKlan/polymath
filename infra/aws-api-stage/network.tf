resource "aws_vpc" "ohio" {
  provider             = aws.ohio
  cidr_block           = "10.42.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags                 = merge(local.tags, { Name = "polymath-ohio" })
}

resource "aws_internet_gateway" "ohio" {
  provider = aws.ohio
  vpc_id   = aws_vpc.ohio.id
  tags     = merge(local.tags, { Name = "polymath-ohio" })
}

resource "aws_subnet" "ohio_public" {
  provider                = aws.ohio
  count                   = 2
  vpc_id                  = aws_vpc.ohio.id
  cidr_block              = cidrsubnet(aws_vpc.ohio.cidr_block, 8, count.index)
  availability_zone       = data.aws_availability_zones.ohio.names[count.index]
  map_public_ip_on_launch = true
  tags                    = merge(local.tags, { Name = "polymath-ohio-public-${count.index + 1}" })
}

resource "aws_subnet" "ohio_database" {
  provider          = aws.ohio
  count             = 2
  vpc_id            = aws_vpc.ohio.id
  cidr_block        = cidrsubnet(aws_vpc.ohio.cidr_block, 8, 10 + count.index)
  availability_zone = data.aws_availability_zones.ohio.names[count.index]
  tags              = merge(local.tags, { Name = "polymath-ohio-db-${count.index + 1}" })
}

resource "aws_route_table" "ohio_public" {
  provider = aws.ohio
  vpc_id   = aws_vpc.ohio.id
  tags     = merge(local.tags, { Name = "polymath-ohio-public" })
}

resource "aws_route" "ohio_internet" {
  provider               = aws.ohio
  route_table_id         = aws_route_table.ohio_public.id
  destination_cidr_block = "0.0.0.0/0"
  gateway_id             = aws_internet_gateway.ohio.id
}

resource "aws_route_table_association" "ohio_public" {
  provider       = aws.ohio
  count          = 2
  subnet_id      = aws_subnet.ohio_public[count.index].id
  route_table_id = aws_route_table.ohio_public.id
}

resource "aws_vpc" "singapore" {
  provider             = aws.singapore
  cidr_block           = "10.43.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags                 = merge(local.tags, { Name = "polymath-singapore" })
}

resource "aws_internet_gateway" "singapore" {
  provider = aws.singapore
  vpc_id   = aws_vpc.singapore.id
  tags     = merge(local.tags, { Name = "polymath-singapore" })
}

resource "aws_subnet" "singapore_public" {
  provider                = aws.singapore
  count                   = 2
  vpc_id                  = aws_vpc.singapore.id
  cidr_block              = cidrsubnet(aws_vpc.singapore.cidr_block, 8, count.index)
  availability_zone       = data.aws_availability_zones.singapore.names[count.index]
  map_public_ip_on_launch = true
  tags                    = merge(local.tags, { Name = "polymath-singapore-public-${count.index + 1}" })
}

resource "aws_route_table" "singapore_public" {
  provider = aws.singapore
  vpc_id   = aws_vpc.singapore.id
  tags     = merge(local.tags, { Name = "polymath-singapore-public" })
}

resource "aws_route" "singapore_internet" {
  provider               = aws.singapore
  route_table_id         = aws_route_table.singapore_public.id
  destination_cidr_block = "0.0.0.0/0"
  gateway_id             = aws_internet_gateway.singapore.id
}

resource "aws_route_table_association" "singapore_public" {
  provider       = aws.singapore
  count          = 2
  subnet_id      = aws_subnet.singapore_public[count.index].id
  route_table_id = aws_route_table.singapore_public.id
}

resource "aws_vpc_peering_connection" "regional" {
  provider    = aws.ohio
  vpc_id      = aws_vpc.ohio.id
  peer_vpc_id = aws_vpc.singapore.id
  peer_region = "ap-southeast-1"
  auto_accept = false
  tags        = merge(local.tags, { Name = "polymath-ohio-singapore" })
}

resource "aws_vpc_peering_connection_accepter" "regional" {
  provider                  = aws.singapore
  vpc_peering_connection_id = aws_vpc_peering_connection.regional.id
  auto_accept               = true
  tags                      = merge(local.tags, { Name = "polymath-ohio-singapore" })
}

resource "aws_route" "ohio_to_singapore" {
  provider                  = aws.ohio
  route_table_id            = aws_route_table.ohio_public.id
  destination_cidr_block    = aws_vpc.singapore.cidr_block
  vpc_peering_connection_id = aws_vpc_peering_connection.regional.id
  depends_on                = [aws_vpc_peering_connection_accepter.regional]
}

resource "aws_route_table" "ohio_database" {
  provider = aws.ohio
  vpc_id   = aws_vpc.ohio.id
  tags     = merge(local.tags, { Name = "polymath-ohio-database" })
}

resource "aws_route_table_association" "ohio_database" {
  provider       = aws.ohio
  count          = 2
  subnet_id      = aws_subnet.ohio_database[count.index].id
  route_table_id = aws_route_table.ohio_database.id
}

resource "aws_route" "database_to_singapore" {
  provider                  = aws.ohio
  route_table_id            = aws_route_table.ohio_database.id
  destination_cidr_block    = aws_vpc.singapore.cidr_block
  vpc_peering_connection_id = aws_vpc_peering_connection.regional.id
  depends_on                = [aws_vpc_peering_connection_accepter.regional]
}

resource "aws_route" "singapore_to_ohio" {
  provider                  = aws.singapore
  route_table_id            = aws_route_table.singapore_public.id
  destination_cidr_block    = aws_vpc.ohio.cidr_block
  vpc_peering_connection_id = aws_vpc_peering_connection.regional.id
  depends_on                = [aws_vpc_peering_connection_accepter.regional]
}
