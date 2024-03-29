{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :no-members:
   :no-inherited-members:
   :no-special-members:
   :show-inheritance:

   {% block methods %}
   .. HACK -- the point here is that we don't want this to appear in the output,
      but the autosummary should still generate the pages.
      .. autosummary::
         :toctree:
      {% for item in all_methods %}
         {%- if not item.startswith('_') or item in ['__call__', '__getitem__'] %}
            {{ name }}.{{ item }}
         {%- endif -%}
      {%- endfor %}
      {% for item in inherited_members %}
         {%- if item in ['__call__', '__getitem__'] %}
         {{ name }}.{{ item }}
         {%- endif -%}
      {%- endfor %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. HACK -- the point here is that we don't want this to appear in the output,
      but the autosummary should still generate the pages.
      .. autosummary::
         :toctree:
      {% for item in attributes %}
         ~{{ name }}.{{ item }}
      {%- endfor %}
   {% endif %}
   {% endblock %}
