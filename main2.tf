# Instancia Original
module "my_ec2_instance" {
  source = "git::https://github.com/fundacionsantafe/module-terraform-ec2-aws.git//?ref=90b14bc9df450ebd58cf528923d43bec8ef8d13b"

  client        = var.client
  project       = var.project
  environment   = var.environment
  function      = var.function

  ami_id        = "ami-0953476d60561c955"
  instance_type = "r6i.large"
  subnet_id     = data.aws_subnet.dmz_1.id
  vpc_id        = data.aws_vpc.vpc_fsfb.id
  key_name      = "" #si agrega un nombre se le creara con SSHKEY
  user_data     = var.user_data


  create_iam_role = true
  managed_policy_arns = [
    "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess",
    "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy",
    "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
  ]
  #inline_policies = {}   # si requiere una politica personalizada

  ingress_rules = [
    {
      protocol         = "tcp"
      from_port        = 5432
      to_port          = 5432
      cidr_blocks      = ["0.0.0.0/0"]
      description      = "allow"
    },
    {
      protocol         = "tcp"
      from_port        = 80
      to_port          = 80
      cidr_blocks      = ["0.0.0.0/0"]
      description      = "allow"
    },
    {
      protocol         = "tcp"
      from_port        = 8080
      to_port          = 8080
      cidr_blocks      = ["0.0.0.0/0"]
      description      = "allow"
    },
    {
      protocol         = "tcp"
      from_port        = 6311
      to_port          = 6311
      cidr_blocks      = ["0.0.0.0/0"]
      description      = "allow"
    }
  ]
  egress_rules = [
    {
      description = "Allow all outbound traffic"
      from_port   = 0
      to_port     = 0
      protocol    = "-1"
      cidr_blocks = ["0.0.0.0/0"]
    },
  ]

  root_volume_size =  50   # Sí desea alterar el tamaño del volumen raiz
  #root_volume_type = "gp3" # Sí desea alterar el tipo del volumen raiz
  #root_kms_key_id  = ""    # ARN si desea una KMS diferente al default

  ebs_volumes = []  # Volumenes externos

}


#########################################################
# Migración a zona privada
#########################################################

module "lb" {
  source = "git::https://github.com/fundacionsantafe/module-terraform-lb-aws.git//?ref=2a668bf1db6084ee9344db7dbbf6cb5a67545609"

  # Variables de nombramiento
  client        = var.client
  project       = var.project
  environment   = var.environment
  functionality = var.function2

  load_balancer_type = "application"  
  subnets            = ["${data.aws_subnet.pres_1.id}", "${data.aws_subnet.pres_2.id}"]
  vpc_id             = data.aws_vpc.vpc_fsfb.id
  internal = true
  
  lb_sg_ingress_rules = [
    {
      from_port   = 80
      to_port     = 80
      protocol    = "tcp"
      cidr_blocks = ["0.0.0.0/0"]
    },
    {
      from_port   = 443
      to_port     = 443
      protocol    = "tcp"
      cidr_blocks = ["0.0.0.0/0"]
    },
  ]

  lb_sg_egress_rules = [
    {
      from_port   = 0
      to_port     = 0
      protocol    = "-1"
      cidr_blocks = ["0.0.0.0/0"]
    }
  ]

  target_type = "instance" # puede ser ip, instance, lambda
  target_port = "80"
  target_protocol = "HTTP"
  targets = [{ id = module.my_ec2_instance2.instance_id, port = 80 }]

  enable_https    = true
  certificate_arn = data.aws_acm_certificate.aliscom.arn
}

module "my_ec2_instance2" {
  source = "git::https://github.com/fundacionsantafe/module-terraform-ec2-aws.git//?ref=90b14bc9df450ebd58cf528923d43bec8ef8d13b"

  client        = var.client
  project       = var.project
  environment   = var.environment
  function      = var.function2

  ami_id        = "ami-0118440887e0079db"
  instance_type = "r6i.large"
  subnet_id     = data.aws_subnet.lan_1.id
  vpc_id        = data.aws_vpc.vpc_fsfb.id
  key_name      = "" #si agrega un nombre se le creara con SSHKEY
  user_data     = var.user_data2


  create_iam_role = true
  managed_policy_arns = [
    "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess",
    "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy",
    "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
  ]
  #inline_policies = {}   # si requiere una politica personalizada

  ingress_rules = [
    {
      protocol         = "tcp"
      from_port        = 5432
      to_port          = 5432
      cidr_blocks      = ["0.0.0.0/0"]
      description      = "allow"
    },
    {
      protocol         = "tcp"
      from_port        = 80
      to_port          = 80
      cidr_blocks      = ["0.0.0.0/0"]
      description      = "allow"
    },
    {
      protocol         = "tcp"
      from_port        = 8080
      to_port          = 8080
      cidr_blocks      = ["0.0.0.0/0"]
      description      = "allow"
    },
    {
      protocol         = "tcp"
      from_port        = 6311
      to_port          = 6311
      cidr_blocks      = ["0.0.0.0/0"]
      description      = "allow"
    }
  ]
  egress_rules = [
    {
      description = "Allow all outbound traffic"
      from_port   = 0
      to_port     = 0
      protocol    = "-1"
      cidr_blocks = ["0.0.0.0/0"]
    },
  ]
  root_volume_size =  50   # Sí desea alterar el tamaño del volumen raiz
  #root_volume_type = "gp3" # Sí desea alterar el tipo del volumen raiz
  #root_kms_key_id  = ""    # ARN si desea una KMS diferente al default
  ebs_volumes = []  # Volumenes externos
}
