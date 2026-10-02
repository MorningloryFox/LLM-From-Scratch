# Cache KV

Na geração autoregressiva, cada novo token depende dos anteriores. Para prever o token seguinte, a atenção precisa das chaves (K) e valores (V) dos tokens anteriores.

## Sem cache

Uma implementação simples reenvia toda a sequência ao modelo depois de cada token. Isso recalcula projeções K e V de posições antigas. O gerador do Feneco faz isso e limita a entrada à janela de contexto. É mais fácil de entender, porém trabalho repetido cresce conforme a saída fica longa.

## Com cache

Durante uma geração, guardamos K e V que cada camada já calculou. Na próxima etapa, processamos apenas o novo token, anexamos seus K/V ao cache e atendemos à sequência acumulada. Os pesos do modelo não mudam; o cache é estado temporário da sessão.

## Custos e cuidados

O cache reduz recálculo, mas usa memória proporcional ao número de camadas, cabeças, posições e dimensão da cabeça. É específico à sequência e ao checkpoint/configuração. Precisamos limpar ou separar caches entre prompts, respeitar limites de contexto e posicionar RoPE corretamente se estiver habilitado.

O Feneco ainda não tem KV-cache. Primeiro vamos validar geração simples e registrar tempo e memória; depois implementaremos o cache e compararemos a mesma geração com e sem ele.
