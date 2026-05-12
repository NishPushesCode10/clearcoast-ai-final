# ===== variables.tf =====
variable "location" {
  description = "Azure region"
  default     = "centralindia"     # Changed to match your Resource Group
}

variable "resource_group_name" {
  description = "Resource Group name"
  default     = "rg-chennai-coastal"
}

variable "app_service_plan_name" {
  description = "App Service Plan name"
  default     = "plan-clearcoast-final"
}

variable "app_service_name" {
  description = "Web App name"
  default     = "clearcoast-ai-final-v2"
}

variable "tags" {
  description = "Common tags"
  default = {
    project     = "clearcoast-ai"
    environment = "dev"
    owner       = "student"
  }
}
