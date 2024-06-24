classdef ControlFunctions
    methods(Static)
        
        function K = lqr(A, B, Q, R)
            % M�todo para calcular o regulador quadr�tico linear (LQR)
            [K, ~] = lqr(A, B, Q, R);
        end

        function plotResponse(sys)
            % M�todo para plotar a resposta do sistema
            [y, t] = step(sys);
            figure;
            plot(t, y);
            title('Resposta ao Degrau do Sistema');
            xlabel('Tempo (s)');
            ylabel('Resposta');
            grid on;
        end
        
        function sys = createSS(A, B, C, D)
            % M�todo para criar um sistema de espa�o de estados
            sys = ss(A, B, C, D);
        end

        function [t, response] = openLoopResponse(amplitude, timeEnd)
            % M�todo para simular uma resposta de la�o aberto hipot�tica
            t = linspace(0, timeEnd, 100);
            response = amplitude * (1 - exp(-t));
        end
        
        function plotPolesAndZeros(sys)
            % M�todo para plotar polos e zeros do sistema
            figure;
            pzmap(sys);
            title('Mapa de Polos e Zeros');
            grid on;
        end

        function result = exemploMetodo(param1, param2)
            % M�todo de exemplo que soma dois par�metros
            result = param1 + param2;
        end
    end
end
