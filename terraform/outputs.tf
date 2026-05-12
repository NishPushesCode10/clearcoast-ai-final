# ===== outputs.tf =====
output "resource_group_name" {
  value = azurerm_resource_group.rg.name
}

output "app_service_name" {
  value = azurerm_linux_web_app.app.name
}

output "app_service_url" {
  value = "https://${azurerm_linux_web_app.app.default_hostname}"
}

output "application_insights_key" {
  value     = azurerm_application_insights.insights.instrumentation_key
  sensitive = true
}

output "app_service_plan_name" {
  value = azurerm_service_plan.plan.name
}
